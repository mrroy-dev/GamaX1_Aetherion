# GamaX1 — First Working Version of Aetherion

GamaX1 is the first real, trainable NLP implementation of the **Aetherion**
architecture. It is built in PyTorch, runs on Google Colab (CPU or GPU), and
translates each *validated* mechanism from the research report into working
code — not the architecture's original theoretical form, but the version
that was actually tested, debugged, and shown to work.

This README reflects the current codebase (see `CHANGES_v2.md` and
`BUGFIXES.md` alongside this file for the detailed history of how it got
here).

## What's in here

```
gamax1/
├── gamax1/
│   ├── layers.py        # Sparse superposition, dynamic sparsity, PTM, hex influence, Router/Validator
│   ├── model.py          # GamaX1Block + GamaX1Model (full causal LM)
│   ├── tokenizer.py       # Character, word, and byte-level BPE tokenizers (4 reserved special tokens)
│   ├── bulk_corpus.py     # Multi-source bulk corpus encoder, per-source content format, persistent cache
│   ├── train.py          # Training script (CLI, metrics, scheduling, resume)
│   ├── compare_dense.py  # Matched sparse-versus-dense experiment
│   └── generate.py       # Text generation script (CLI, incl. --chat mode)
├── prepare_large_corpus.py  # Gutenberg book downloader/combiner (curated or range presets)
├── compare_dense.py       # Convenience entry point
├── GamaX1_Colab_Training_v2.ipynb  # Google Colab notebook
├── requirements.txt        # Single dependency: torch
├── CHANGES_v2.md           # What changed in the special-token/per-source-format revision, and why
├── BUGFIXES.md             # 4 verified-and-fixed correctness bugs
└── setup.py
```

## Architecture: how this maps to Aetherion's four pillars

Aetherion's design frame is four pillars — Adaptive Superposition, Dynamic
Resource Allocation, Hierarchical/curriculum-style learning, and Embedded
Safety — plus a proposed fifth, Q-Spark Attention (an attention-mechanism
replacement, still a separate research thread, not implemented here).
Being direct about which of these are actually active in GamaX1's *trained*
model, versus merely present in the code, matters more than the mapping
table looking complete:

| Pillar / mechanism | Code | Status |
|---|---|---|
| **Adaptive Superposition** — sparse superposition FFN, matching ~96–98% of dense accuracy at a fraction of the compute | `layers.SparseSuperpositionLinear` | **Fully active.** Used as the FFN in every block, drives training the entire time. |
| **Adaptive Superposition** — sparsity level gated on loss *trend*, never instantaneous signal | `layers.DynamicSparsityController` | **Fully active.** See the real run data below for `k` shrinking from 1536→384 over training. |
| Dead-feature prevention (asymmetric entry/exit hysteresis) | `layers.ProbationaryMemoryTracker` | **Active during training** — a training-stability mechanism. Its forced-activation nudge previously had its gradient silently cut by a stray `.detach()` (see `BUGFIXES.md` #3); fixed. |
| **Dynamic Resource Allocation** — trained layer-skip router | `layers.RouterExpert` | **Constructed, not trained or used.** Never invoked in `forward`/`generate`; receives no gradient today. A reserved hook, not a working mechanism. |
| Early-exit gating by answer *stability*, not raw confidence | `layers.ValidatorExpert.is_stable` | **Active, inference-time only**, behind `--hierarchical_exit`. Runs each block exactly once per token (incremental depth), never more expensive than the full dense path. |
| **Hierarchical/curriculum-style learning** | — | **Not implemented.** No staged/difficulty-ordered training schedule exists; the Router/Validator "hierarchical exit" is a different thing (inference-time depth-adaptivity, not a training curriculum). |
| **Embedded Safety** | — | **Not implemented.** A genuine safety/conscience layer for Aetherion (codename Syneidesis) is a separate, still-theoretical research paper — nothing in this codebase implements it. |
| Hexagonal neighbor influence | `layers.HexNeighborInfluence` | Included, **off by default** — only helps when hidden features cluster at the right scale, which can't be guaranteed for a language model; enable and evaluate empirically, don't assume a benefit. |

**Attention is retained.** Aetherion's validated claim is about replacing a
dense *feed-forward* layer with sparse superposition — never tested as an
attention replacement. GamaX1 uses standard causal self-attention for
token-mixing and Aetherion's sparse mechanism for the FFN.

## Tokenizer: 4 reserved special tokens

`BPETokenizer` reserves four ids right after the last BPE merge, all
derived from `len(self.merges)` (no serialization change — an existing
`tokenizer.json` still loads and gets all four automatically):

| Token | Accessor | Purpose |
|---|---|---|
| `<\|eos\|>` | `tok.eos_id` | Document / conversation-turn boundary |
| `<\|user\|>` | `tok.user_id` | Start of a user turn (only for sources with genuinely known roles) |
| `<\|assistant\|>` | `tok.assistant_id` | Start of an assistant turn |
| `<\|pad\|>` | `tok.pad_id` | Reserved for a future instruction-tuning stage; **not used** in bulk pretraining (every training window is a fixed-length slice of one continuous stream, so nothing needs padding yet) |

`tok.vocab_size` = `256 + merges + 4`. To hit an exact total (e.g. 16,000),
pass `--bpe_vocab_size <target - 4>`. `tok.decode_with_boundaries(ids)`
renders each special token as a readable tag instead of dropping it —
useful for inspecting a cache or a generation.

## Bulk corpus: per-source content format

Different source folders hold genuinely different kinds of text, and a
single plain `eos_id`-between-files boundary isn't enough to stop a model
from drifting into an unrelated pattern mid-generation — a file can also
glue several unrelated exchanges together internally, or come from a
license-boilerplate-heavy source. `bulk_corpus.py` assigns each source
folder one of three formats (`DEFAULT_SOURCE_FORMATS`), chosen
deliberately per source rather than guessed from content:

| Format | Default sources | What happens |
|---|---|---|
| `prose` | `books`, `books2`, `wiki` | Project Gutenberg license boilerplate stripped (same START/END markers as `prepare_large_corpus.py`); whole file is one `eos_id`-bounded unit. |
| `user_assistant` | `Conversations-200k`, `Discord-Dialogues` | Split first on a standalone `---` line (Discord files glue several unrelated exchanges together this way); within each resulting exchange, literal `User:`/`Assistant:` labels are replaced with the real `<\|user\|>`/`<\|assistant\|>` ids; `eos_id` between every exchange, not just every file. |
| `generic_turns` | `Reddit-Constructive` | Same `---`-splitting, but **no role tags are fabricated** — `Speaker 0:`/`Speaker 1:` don't reliably mean the same role across threads, so guessing would be a worse signal than no tag. Still `eos_id`-bounded per turn. |

A source not in this dict falls back to `prose` with a one-time warning.
Pass your own `source_formats={...}` to `build_or_load_bulk_tokens()` to
override without editing the file.

File identity for resume/change-detection is size-only (not mtime — Google
Drive's FUSE mount doesn't reliably preserve mtimes across a remount, which
previously made a fully-synced cache look "changed" on every fresh Colab
session). `--rebuild_bulk_cache` quarantines old cache files with a `.bak`
suffix rather than deleting them.

## CLI usage

```bash
# Train on the bundled corpus
python -m gamax1.train --max_steps 2000

# Train on your own text file
python -m gamax1.train --data path/to/your.txt --max_steps 5000 --n_layers 6

# Byte-level BPE (recommended for real corpora: no <unk>, more tokens/byte)
python -m gamax1.train --tokenizer bpe --data path/to/your.txt --max_steps 5000 --bpe_vocab_size 8000

# --eval_batch_size now defaults to --batch_size (not a fixed 128) -- a
# bigger eval batch running right after a training step, while its memory
# is still reserved, was a common CUDA OOM trap. Override explicitly only
# once you've confirmed the GPU has headroom:
python -m gamax1.train --batch_size 16 --eval_batch_size 32

# Train on a directory of source folders (books/, wiki/, Conversations-200k/, ...)
# First run: builds the memory-mapped token cache. Later runs reuse it.
python -m gamax1.train --data_dir data/ --tokenizer bpe \
  --bulk_cache_dir data/bulk_cache --bpe_vocab_size 16000 --max_steps 30000 \
  --out_dir checkpoints/run1

# Force a full re-encode (needed after changing source_formats, the
# tokenizer's special tokens, or the source files/BPE settings)
python -m gamax1.train --data_dir data/ --tokenizer bpe \
  --bulk_cache_dir data/bulk_cache --rebuild_bulk_cache --bpe_vocab_size 16000 \
  --max_steps 30000 --out_dir checkpoints/run1_v2

# Resume (auto-restores optimizer, sparsity-controller trend history, and
# every layer's dead-feature-prevention state -- not just the model weights)
python -m gamax1.train --data_dir data/ --tokenizer bpe \
  --bulk_cache_dir data/bulk_cache --resume_from checkpoints/run1/gamax1_latest.pt \
  --max_steps 30000 --out_dir checkpoints/run1

# Plain completion
python -m gamax1.generate --ckpt checkpoints/run1/gamax1_latest.pt \
  --prompt "Once upon a time" --max_new_tokens 300 --repetition_penalty 1.2

# Stop at a genuine document/turn boundary instead of always running to max_new_tokens
python -m gamax1.generate --ckpt checkpoints/run1/gamax1_latest.pt --prompt "..." --stop_at_eos

# Chat-style: wraps the prompt as <|user|> ... <|assistant|> using the real
# reserved ids and generates the reply, stopping at eos_id automatically.
# Only meaningful on a checkpoint trained with the user_assistant source format.
python -m gamax1.generate --ckpt checkpoints/run1/gamax1_latest.pt \
  --chat --prompt "What is the capital of France?"

# Hierarchical early exit at inference time (Validator-gated, incremental depth)
python -m gamax1.generate --ckpt checkpoints/run1/gamax1_latest.pt --hierarchical_exit

# Download/combine Gutenberg books for a quick single-file corpus
python prepare_large_corpus.py --preset curated   # 93 hand-picked long novels
python prepare_large_corpus.py --preset range_1_400   # plain IDs 1-400 (default)
python prepare_large_corpus.py --ids 2600 100 1400 2701   # your own list
```

`train` logs train/validation loss, perplexity, current learning rate, and
sparse active-unit compute at each evaluation interval.
`Corpus/model size check` compares tokens to trainable parameters; aim for
at least **10 tokens per parameter** (`--min_tokens_per_param 10`) as a
rough guardrail against memorization, not a guarantee of generalization.

## Real training run (multi-source bulk corpus)

A concrete data point, from an actual run on a combined conversational/
forum/book corpus (5 sources, `books2` excluded from this particular run):
21,645 files, 1,542,490,585 tokens — Conversations-200k 268.3M,
Discord-Dialogues 234.9M, Reddit-Constructive 265.2M, books 740.0M,
wiki 34.1M. Paired with a `d_model=768, n_heads=12, n_layers=12,
n_features=3072` model (97,737,985 parameters, ~15.8 tokens/parameter),
single T4 GPU, `--batch_size 16`, mixed precision:

| Step | train_loss | val_loss | val_ppl | sparsity_k | compute_ratio_vs_dense |
|---|---|---|---|---|---|
| 1 | 9.89 | 9.92 | 18,920 | 1536 | 2.00x |
| 1,000 | 4.95 | 5.25 | 191 | 1006 | 3.05x |
| 3,000 | 4.29 | 4.33 | 76 | **384 (floor)** | **8.00x** |
| 10,000 | 3.42 | 3.48 | 33 | 384 | 8.00x |
| 20,000 (separate/expanded-vocab run) | 3.92 | 3.82 | — | — | — |

`sparsity_k` hit its configured floor (`k_min = n_features/8 = 384`) around
step 3,000 and stayed there; `compute_ratio_vs_dense` is *dense units ÷
active units* — 8.00x means the model runs at **one-eighth** the compute of
an equivalent dense FFN, which is the efficiency mechanism working as
intended, not rising overhead.

**Generation at ~33% of a planned 30,000-step run (val_ppl ≈ 32) showed two
distinct symptoms**, since fixed by different changes:
- *Repetition loops* (e.g. "king of the king of the world..." from an
  unrelated prompt, echoing an over-represented `books` pattern) —
  `--repetition_penalty 1.2–1.3` largely addresses this.
- *Format/topic hijacking mid-generation* (drifting into `User:`/
  `Assistant:` or `---`-separated Reddit-style turns regardless of the
  prompt) — this is exactly what the per-source content format / `eos_id`
  boundary fix targets; it requires re-encoding the corpus with the current
  `bulk_corpus.py` to take effect (an older cache/checkpoint never saw
  these boundaries during training).

Separately, basic arithmetic ("what is 2 plus 2?") and precise factual
recall (e.g. a specific country's highest peak) were **not** reliably
produced at this scale/stage — expected: no dedicated math data in the
corpus, and next-token pretraining on mixed-domain text does not by itself
teach reliable instruction-following or fact recall; see Roadmap.

## Avoiding memorization

`--min_tokens_per_param` (default 10) and `--perplexity_memorization_floor`
(default 1.5) together catch the classic "generalizing" pattern (val rises
while train falls) and the subtler one (both fall together on too little
data per parameter) — see `--early_stop_on_overfit`. `--max_vocab_size`
caps a word-level vocabulary; GamaX1 ties input/output embedding weights,
so it doesn't pay for two vocab-sized matrices. `--auto_size_model` picks a
viable architecture for a given corpus without producing a degenerate one.

## Known, fixed correctness bugs

See `BUGFIXES.md` for full detail. Summary: `get_batch`'s sampling bound
was off by one (crashed on the smallest valid corpus, never sampled the
final valid window); batched repetition penalty used only sequence 0's
tokens for every row (latent — the CLI always uses batch=1); the
dead-feature-prevention nudge had its gradient accidentally cut by a stray
`.detach()`; checkpoint resume silently lost the sparsity controller's
trend history and every layer's dead-feature-tracking state. All four are
fixed, reproduced-then-verified in sandbox, with the fix.

## Roadmap / not yet in this version

- **Router training** — needs a difficulty-labeling scheme (e.g. a
  shallow-vs-full-depth loss comparison, or a per-token loss threshold)
  before `RouterExpert` can be trained end-to-end; currently unused.
- **Curriculum/hierarchical-style training** — no staged or
  difficulty-ordered training schedule is implemented yet.
- **Syneidesis (Embedded Safety)** — a separate, still-theoretical
  moral-reasoning veto-layer paper; not implemented here.
- **Q-Spark Attention** — a proposed original attention-mechanism
  replacement, developed as an independent side research thread; GamaX1
  keeps standard causal self-attention until/unless it's shown better.
- **Supervised instruction-tuning stage** — the `<|user|>`/`<|assistant|>`
  tags make answer-only loss masking possible to add, but `train.py`
  currently trains on plain next-token prediction across the whole tagged
  stream, not masked to the assistant's tokens only.
- **Arithmetic / precise factual recall** — needs dedicated data and/or
  more scale; a tokenizer or corpus-format fix does not substitute for this.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

## License

Research/educational use. No warranty. This is a first-version ("v1")
prototype accompanying an ongoing research report — expect rough edges.