"""
gamax1/model.py
================
GamaX1: first working version of Aetherion as a real, trainable NLP
language model.

Architecture note (honest design choice): the Aetherion research
report's core validated claim is about REPLACING A DENSE FEED-FORWARD
LAYER with a sparse-superposition layer at large-but-sparse width,
retaining ~96-98% accuracy at a fraction of the compute (Sections 5.1,
5.12). It was not tested as a replacement for attention/token-mixing.
GamaX1 therefore uses standard causal multi-head self-attention for
sequence/token mixing (attention is a well-established, necessary
mechanism for language modeling that Aetherion was never proposed as a
replacement for) and substitutes the Transformer's usual dense FFN
with an AetherionFFN block (SparseSuperpositionLinear + dynamic
sparsity + probationary memory). This is the most defensible way to
bring the *validated* Aetherion contributions into a real NLP model
without overclaiming mechanisms that were never tested at this task.

Router/Validator experts (Section 5.4/5.6/5.11) are wired in as an
INFERENCE-TIME compute-saving option (`use_hierarchical_exit=True` in
`generate`), since dynamic per-sample depth is straightforward at
inference (sequential decoding) but would require complex ragged-batch
handling to train efficiently -- a limitation stated plainly rather
than hidden.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    SparseSuperpositionLinear,
    DynamicSparsityController,
    ProbationaryMemoryTracker,
    RouterExpert,
    ValidatorExpert,
)


def apply_repetition_penalty(logits: torch.Tensor, present: torch.Tensor, penalty: float) -> torch.Tensor:
    """Suppress tokens already present in the sequence during sampling.

    ``logits``: (batch, vocab); ``present``: (batch, vocab) bool mask, True
    where that batch row's own sequence already contains the token -- or a
    (vocab,) mask applied identically to every row, for a single shared
    sequence. Positive logits are divided by ``penalty`` and negative ones
    are multiplied by it, so both directions push the token's probability
    down. A penalty of 1.0 is the identity. Uses ``torch.where`` (elementwise,
    broadcasting-safe) rather than boolean fancy-indexing, so a per-row
    ``present`` mask is applied per-row rather than collapsing the batch.
    """
    if penalty == 1.0:
        return logits
    scaled = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(present, scaled, logits)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)).view(
            1, 1, max_seq_len, max_seq_len
        )
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_heads, T, head_dim)

        # SDPA selects FlashAttention or the memory-efficient CUDA kernel
        # when available, avoiding materializing the full attention matrix.
        # This is important for long contexts and larger training batches.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self.causal_mask[:, :, :T, :T],
            dropout_p=self.dropout.p if self.training else 0.0,
        )  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class AetherionFFN(nn.Module):
    """Drop-in replacement for a Transformer block's dense FFN, using
    the validated sparse-superposition mechanism instead of a dense
    hidden layer."""

    def __init__(self, d_model: int, n_features: int, dropout: float = 0.1,
                 hex_influence: bool = False):
        super().__init__()
        self.sparse = SparseSuperpositionLinear(d_model, n_features, hex_influence=hex_influence)
        self.ptm = ProbationaryMemoryTracker(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, k: int, use_ptm: bool = True):
        nudge = self.ptm.nudge_indices() if use_ptm else None
        out = self.sparse(x, k=k, nudge_indices=nudge)
        if use_ptm and self.training:
            self.ptm.update(self.sparse.last_active_mask)
        return self.dropout(out)


class DenseFFN(nn.Module):
    """Dense control FFN with the same projection shapes as AetherionFFN.

    It exists solely to make sparse-versus-dense experiments fair: both paths
    have the same feature width and parameter count, but this path evaluates
    every hidden feature for every token.
    """

    def __init__(self, d_model: int, n_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, n_features)
        self.out_proj = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, k: int = None, use_ptm: bool = True):
        return self.dropout(self.out_proj(F.relu(self.in_proj(x))))


class GamaX1Block(nn.Module):
    def __init__(self, d_model, n_heads, n_features, max_seq_len, dropout=0.1,
                 hex_influence=False, dense_mode=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = (DenseFFN(d_model, n_features, dropout=dropout) if dense_mode else
                    AetherionFFN(d_model, n_features, dropout=dropout,
                                 hex_influence=hex_influence))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, k: int, use_ptm: bool = True):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x), k=k, use_ptm=use_ptm))
        return x


class GamaX1Model(nn.Module):
    """First version ("GamaX1") of Aetherion as a real causal language
    model. See module docstring for the architecture's relationship to
    the research report's validated findings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        n_features: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        hex_influence: bool = False,
        sparsity_k_init: int = 256,
        sparsity_k_min: int = 64,
        dense_mode: bool = False,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_features = n_features
        self.dense_mode = dense_mode
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GamaX1Block(d_model, n_heads, n_features, max_seq_len, dropout,
                         hex_influence, dense_mode)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.sparsity_ctrl = DynamicSparsityController(
            k_init=sparsity_k_init, k_min=sparsity_k_min, k_max=n_features,
        )
        self.router = RouterExpert(d_model)
        self.validator = ValidatorExpert(patience=1)

        self.apply(self._init_weights)
        # Standard language-model weight tying: input and output vocabulary
        # representations share one matrix, avoiding a second vocab-sized
        # parameter block that otherwise dominates small models with large
        # word vocabularies.
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, k: int = None, use_ptm: bool = True):
        B, T = idx.shape
        assert T <= self.max_seq_len, "sequence length exceeds max_seq_len"
        k = k if k is not None else self.sparsity_ctrl.k

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.dropout(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x, k=k, use_ptm=use_ptm)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # Standard categorical cross-entropy for next-token prediction.
            # (Deliberately NOT the pairwise ranking loss used elsewhere in
            # the research report -- see module docstring / layers.py.)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def active_units_per_token(self, k: int = None) -> int:
        """Compute-accounting helper: how many sparse hidden units are
        actually evaluated per token, summed across all blocks --
        directly comparable to a dense baseline's fixed n_features."""
        if self.dense_mode:
            return self.n_features * len(self.blocks)
        k = k if k is not None else self.sparsity_ctrl.k
        return k * len(self.blocks)

    def ptm_state_dicts(self):
        """List of each block's ProbationaryMemoryTracker state, in block
        order (``None`` for a dense-mode block, which has no PTM). PTM
        objects are plain Python objects, not nn.Modules, so they are
        NOT captured by ``state_dict()``/``load_state_dict()`` -- without
        explicitly saving/restoring this, a resumed run starts every
        layer's dead-feature tracking from scratch (miss/success counts,
        probation membership all reset), even though the model weights
        themselves resumed correctly."""
        return [
            block.ffn.ptm.state_dict() if hasattr(block.ffn, "ptm") else None
            for block in self.blocks
        ]

    def load_ptm_state_dicts(self, states):
        """Inverse of ``ptm_state_dicts``. Tolerant of a shorter/None list
        (e.g. an older checkpoint saved before this existed) -- restores
        whatever is present and leaves the rest at their freshly-initialized
        state rather than raising."""
        if not states:
            return
        for block, state in zip(self.blocks, states):
            if state is not None and hasattr(block.ffn, "ptm"):
                block.ffn.ptm.load_state_dict(state)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None,
                 repetition_penalty: float = 1.0, use_hierarchical_exit: bool = False,
                 eos_id: int = None):
        """Autoregressive sampling. If use_hierarchical_exit, the Validator
        (Section 5.6/5.11) decides per step whether the stack of blocks
        processed so far is already stable, exiting early instead of always
        running every block -- a genuine, inference-time-only use of the
        hierarchical-exit mechanism (see module docstring for why this isn't
        done at train time). Blocks are run incrementally (each block
        evaluated exactly once per token, carrying its output forward to the
        next depth check) rather than restarting the forward pass from the
        embeddings at every depth, so early exit is never more expensive
        than the full dense path.

        If ``eos_id`` is given and the batch is a single sequence, sampling
        stops as soon as that token is produced (a genuine end-of-document
        signal from the tokenizer -- see tokenizer.py/bulk_corpus.py),
        instead of always running to ``max_new_tokens``. Ignored for
        batch>1, where sequences could finish at different steps and
        per-sequence stopping isn't implemented here.

        Note: `self.router` (RouterExpert) is constructed but not currently
        consulted here -- exit is decided purely by the Validator's
        stability check. Router is a reserved hook for a future trained
        difficulty-prediction signal (Section 5.4/5.11), not yet wired in;
        it receives no gradient today since its output is unused in
        `forward`/`generate`.
        """
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("idx must have shape (batch, sequence) with a non-empty sequence")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")

        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            k = self.sparsity_ctrl.k

            if use_hierarchical_exit:
                self.validator.reset()
                x = self.token_emb(idx_cond) + self.pos_emb(
                    torch.arange(idx_cond.size(1), device=idx_cond.device).unsqueeze(0)
                )
                logits = None
                for n_blocks_used, block in enumerate(self.blocks, start=1):
                    x = block(x, k=k, use_ptm=False)
                    logits = self.head(self.ln_f(x))[:, -1, :]
                    top_id = logits.argmax(dim=-1)
                    if self.validator.is_stable(top_id) and n_blocks_used < len(self.blocks):
                        break
            else:
                logits, _ = self.forward(idx_cond, k=k, use_ptm=False)
                logits = logits[:, -1, :]

            logits = logits / temperature
            if repetition_penalty != 1.0:
                # Per-row mask: (batch, vocab), True where THAT row's own
                # sequence-so-far contains the token -- not just row 0's,
                # so batched generation penalizes each sequence by its own
                # history (previously all rows were penalized using only
                # sequence 0's tokens).
                present = torch.zeros(idx.size(0), logits.shape[-1], dtype=torch.bool, device=logits.device)
                present.scatter_(1, idx, True)
                logits = apply_repetition_penalty(logits, present, float(repetition_penalty))
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None and idx.size(0) == 1 and next_id.item() == eos_id:
                break
        return idx