"""Training utilities and CLI for GamaX1."""

import argparse
from dataclasses import dataclass
import math
import os
import time

import torch

from .model import GamaX1Model
from .tokenizer import BPETokenizer, CharTokenizer, WordTokenizer, word_tokenizer_warning
from .bulk_corpus import build_or_load_bulk_tokens


def perplexity(loss: float) -> float:
    """Return exp(loss), reporting infinity instead of overflowing."""
    loss = float(loss)
    return math.exp(loss) if loss < math.log(float.fromhex("0x1.fffffffffffffp+1023")) else float("inf")


def get_lr_schedule(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup followed by cosine decay to ten percent of base LR."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    warmup_steps = max(0, min(warmup_steps, max_steps))
    if warmup_steps and step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def is_overfitting(val_losses, train_losses, patience: int) -> bool:
    """Detect consecutive validation regression while training still improves."""
    if patience <= 0 or len(val_losses) < patience + 1 or len(train_losses) < patience + 1:
        return False
    recent_val = val_losses[-(patience + 1):]
    recent_train = train_losses[-(patience + 1):]
    return (all(b > a for a, b in zip(recent_val, recent_val[1:])) and
            all(b <= a for a, b in zip(recent_train, recent_train[1:])))


def tokens_per_parameter(token_count: int, parameter_count: int) -> float:
    """Return the corpus-size-to-model-capacity heuristic used by training."""
    if token_count < 0:
        raise ValueError("token_count must not be negative")
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    return token_count / parameter_count


def is_memorization_detected(
    token_count: int,
    parameter_count: int,
    train_loss: float,
    val_loss: float,
    min_tokens_per_param: float = 10.0,
    perplexity_memorization_floor: float = 1.5,
) -> bool:
    """Detect implausibly low loss for a corpus that is too small for the model.

    Unlike ``is_overfitting``, this intentionally does not require validation
    loss to rise: a small validation split drawn from the same tiny corpus can
    be memorized along with training data.
    """
    if min_tokens_per_param < 0:
        raise ValueError("min_tokens_per_param must not be negative")
    if perplexity_memorization_floor <= 0:
        raise ValueError("perplexity_memorization_floor must be positive")
    corpus_ratio = tokens_per_parameter(token_count, parameter_count)
    best_perplexity = min(perplexity(train_loss), perplexity(val_loss))
    return corpus_ratio < min_tokens_per_param and best_perplexity < perplexity_memorization_floor


def create_optimizer(model: GamaX1Model, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Create the training optimizer in one testable place."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


@dataclass(frozen=True)
class AutoSizeResult:
    """Architecture and capacity assessment selected by ``auto_size_model``."""

    d_model: int
    n_heads: int
    n_layers: int
    n_features: int
    parameter_count: int
    tokens_per_param: float
    status: str


def _parameter_count_for_config(
    vocab_size: int, block_size: int, d_model: int, n_heads: int,
    n_layers: int, n_features: int,
) -> int:
    """Count actual trainable parameters for an auto-size candidate."""
    # Candidate construction initializes tensors; preserve the training RNG
    # state so enabling auto-size does not silently change reproducibility.
    with torch.random.fork_rng(devices=[]):
        model = GamaX1Model(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, n_features=n_features, max_seq_len=block_size,
            sparsity_k_init=max(1, n_features // 2),
            sparsity_k_min=max(1, n_features // 8),
        )
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def auto_size_model(
    token_count: int, vocab_size: int, block_size: int, n_heads: int,
    target_tokens_per_param: float = 40.0, min_tokens_per_param: float = 10.0,
) -> AutoSizeResult:
    """Select a viable model, scaling upward only when the token budget permits.

    The hard floor preserves a meaningful attention width and a wide sparse
    feature space. On very small corpora that floor may exceed the recommended
    capacity budget; the correct response is a warning and more data, never a
    degenerate model with one-dimensional heads or a four-unit feature space.
    """
    if token_count <= 0 or vocab_size <= 0 or block_size <= 0:
        raise ValueError("token_count, vocab_size, and block_size must be positive")
    if target_tokens_per_param <= 0 or min_tokens_per_param <= 0:
        raise ValueError("token-per-parameter targets must be positive")

    selected_heads = max(2, n_heads)
    base_d_model = max(64, selected_heads)
    base_d_model = math.ceil(base_d_model / selected_heads) * selected_heads
    # Each tuple increases usable depth/width while keeping n_features at 4x
    # d_model, the minimum wide sparse-superposition regime.
    candidate_shapes = [
        (base_d_model, 2),
        (base_d_model, 3),
        (base_d_model * 2, 3),
        (base_d_model * 2, 4),
        (base_d_model * 4, 4),
        (base_d_model * 4, 6),
        (base_d_model * 8, 6),
    ]
    candidates = []
    target_parameter_budget = token_count / target_tokens_per_param
    minimum_parameter_budget = token_count / min_tokens_per_param
    for d_model, n_layers in candidate_shapes:
        n_features = 4 * d_model
        parameter_count = _parameter_count_for_config(
            vocab_size, block_size, d_model, selected_heads, n_layers, n_features,
        )
        candidates.append((d_model, n_layers, n_features, parameter_count))
        if parameter_count > target_parameter_budget:
            break

    selected = candidates[0]
    if selected[3] <= target_parameter_budget:
        for candidate in candidates:
            if candidate[3] <= target_parameter_budget:
                selected = candidate
            else:
                break

    d_model, n_layers, n_features, parameter_count = selected
    ratio = tokens_per_parameter(token_count, parameter_count)
    if candidates[0][3] > minimum_parameter_budget:
        status = "hard_floor"
    elif ratio < target_tokens_per_param:
        status = "above_minimum_below_target"
    else:
        status = "comfortably_above_target"
    return AutoSizeResult(
        d_model=d_model, n_heads=selected_heads, n_layers=n_layers,
        n_features=n_features, parameter_count=parameter_count,
        tokens_per_param=ratio, status=status,
    )


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str,
    start_indices: torch.Tensor = None,
):
    """Sample next-token windows, optionally from a predefined split."""
    if start_indices is None:
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    else:
        if start_indices.numel() == 0:
            raise ValueError("start_indices must not be empty")
        ix = start_indices[torch.randint(len(start_indices), (batch_size,))]
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    # Bulk caches are int32 to keep the on-disk footprint small. Convert only
    # the sampled batch to the Long dtype required by cross-entropy.
    return x.to(device=device, dtype=torch.long), y.to(device=device, dtype=torch.long)


def split_training_windows(
    data: torch.Tensor, block_size: int, validation_fraction: float = 0.1,
    strategy: str = "random_windows", seed: int = 1337,
):
    """Create either a legacy tail split or disjoint randomized text windows.

    Random windows are block-sized non-overlapping segments sampled throughout
    the corpus. They produce a representative validation distribution for a
    multi-book corpus without leaking individual tokens between train and
    validation windows. ``tail`` remains useful for a deliberately harder
    final-book/domain-shift evaluation.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if strategy == "tail":
        split_at = int((1 - validation_fraction) * len(data))
        return data[:split_at], data[split_at:], None, None
    if strategy != "random_windows":
        raise ValueError(f"unknown validation split strategy: {strategy}")

    starts = torch.arange(0, len(data) - block_size, block_size + 1)
    if len(starts) < 2:
        raise ValueError("corpus is too short for a non-overlapping random-window split")
    generator = torch.Generator().manual_seed(seed)
    shuffled = starts[torch.randperm(len(starts), generator=generator)]
    val_count = max(1, int(round(validation_fraction * len(shuffled))))
    return data, data, shuffled[val_count:], shuffled[:val_count]


def evaluate_loss(
    model: GamaX1Model, data: torch.Tensor, block_size: int, eval_batch_size: int,
    eval_batches: int, device: str, k: int = None, start_indices: torch.Tensor = None,
    use_amp: bool = False,
) -> float:
    """Return the mean loss across independent validation batches.

    A larger sampled token population makes validation metrics substantially
    less sensitive to which rare words happen to appear in one small batch.
    """
    if eval_batches <= 0 or eval_batch_size <= 0:
        raise ValueError("eval_batches and eval_batch_size must be positive")
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            xb, yb = get_batch(data, block_size, eval_batch_size, device, start_indices)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp and device.startswith("cuda")):
                _, loss = model(xb, targets=yb, k=k)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def tokenizer_class(name: str):
    return {"word": WordTokenizer, "bpe": BPETokenizer}.get(name, CharTokenizer)


def checkpoint_dict(model, optimizer, tokenizer, config, step: int, scaler=None):
    """Collect state needed for a robust cross-session training resume."""
    if isinstance(tokenizer, BPETokenizer):
        vocab = None
        merges = tokenizer.merges
    elif isinstance(tokenizer, WordTokenizer):
        vocab = tokenizer.tokens
        merges = None
    else:
        vocab = tokenizer.chars
        merges = None
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sparsity_controller_state": model.sparsity_ctrl.state_dict(),
        "step": step,
        "vocab": vocab,
        "merges": merges,
        "config": config,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(path, model, optimizer, tokenizer, config, step, scaler=None):
    """Atomically save a checkpoint so a Colab interruption cannot leave a half-file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(
        checkpoint_dict(model, optimizer, tokenizer, config, step, scaler=scaler),
        tmp_path,
    )
    os.replace(tmp_path, path)
    print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train GamaX1 on a text corpus.")
    parser.add_argument("--data", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_corpus.txt"))
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Recursive directory of .txt books for bulk training. Uses BPE and a memory-mapped token cache.")
    parser.add_argument("--bulk_cache_dir", type=str, default="data/bulk_cache",
                        help="Directory for the bulk int32 token cache (default: data/bulk_cache).")
    parser.add_argument("--rebuild_bulk_cache", action="store_true",
                        help="Re-encode all books even if the bulk token cache is reusable.")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_features", type=int, default=1024)
    parser.add_argument("--auto_size_model", action="store_true",
                        help="Choose a viable model from corpus token count without going below hard architecture floors. "
                             "A rough safety net; manually size and validate models for serious use.")
    parser.add_argument("--auto_size_target_tokens_per_param", type=float, default=40.0,
                        help="Target corpus tokens per parameter for --auto_size_model (default: 40).")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay regularization (default: 0.01).")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout used by embeddings, attention, FFNs, and residual paths (default: 0.1).")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=10,
                        help="Independent validation batches averaged at each evaluation (default: 10).")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Sequences per validation batch. Defaults to --batch_size if not given -- "
                             "NOT a larger fixed value: an eval batch bigger than the training batch runs "
                             "right after a training step, while its memory is still reserved, and is a "
                             "common cause of a CUDA out-of-memory crash immediately after step 1 succeeds. "
                             "Pass a larger value explicitly only if you've confirmed the GPU has headroom.")
    parser.add_argument("--validation_split", choices=("random_windows", "tail"), default="random_windows",
                        help="Validation split: representative non-overlapping random windows (default) or final corpus tail.")
    parser.add_argument("--validation_split_seed", type=int, default=1337,
                        help="Random-window validation split seed (default: 1337).")
    parser.add_argument("--overfit_patience", type=int, default=3)
    parser.add_argument("--min_tokens_per_param", type=float, default=10.0,
                        help="Warn about memorization below this corpus-token/parameter ratio (default: 10).")
    parser.add_argument("--perplexity_memorization_floor", type=float, default=1.5,
                        help="Flag low-capacity-ratio runs when train or validation perplexity falls below this value (default: 1.5).")
    parser.add_argument("--early_stop_on_overfit", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save a resumable latest checkpoint every N steps (default: 100).")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint to resume. If omitted, automatically resumes checkpoints/gamax1_latest.pt when present.")
    parser.add_argument("--tokenizer", choices=("char", "word", "bpe"), default="char",
                        help="char: character-level; word: word-level with <unk>; "
                             "bpe: dependency-free byte-level BPE (recommended for real corpora).")
    parser.add_argument("--bpe_vocab_size", type=int, default=8000,
                        help="BPE vocabulary size, including the 256 byte tokens (default: 8000).")
    parser.add_argument("--bpe_sample_chars", type=int, default=3_000_000,
                        help="Chars of training text scanned for BPE pair statistics (default: 3000000). "
                             "Larger is slightly better but costs linear Python time.")
    # 15K covers common English across several novels while preventing a huge
    # vocabulary projection from consuming nearly all parameters in small LMs.
    parser.add_argument("--max_vocab_size", type=int, default=15_000,
                        help="Maximum word-tokenizer vocabulary size, including <unk> (default: 15000). "
                             "Ignored by the character and BPE tokenizers.")
    parser.add_argument("--hex_influence", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable CUDA mixed-precision training (uses more GPU memory).")
    args = parser.parse_args()
    if args.eval_batch_size is None:
        # Match the training batch size by default rather than a fixed,
        # potentially much larger value -- see the flag's help text for why
        # a bigger-than-training eval batch is a common CUDA OOM trap.
        args.eval_batch_size = args.batch_size
    if args.data_dir and args.tokenizer != "bpe":
        parser.error("--data_dir requires --tokenizer bpe")
    if args.warmup_steps is None:
        args.warmup_steps = max(100, args.max_steps // 20)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Cross-session default: if the user does not explicitly choose a
    # checkpoint, continue from the latest saved training state when one
    # exists.  This is independent of the corpus token cache.
    latest_checkpoint = os.path.join(args.out_dir, "gamax1_latest.pt")
    resume_path = args.resume_from or (latest_checkpoint if os.path.exists(latest_checkpoint) else None)
    resume_ckpt = (
        torch.load(resume_path, map_location=device, weights_only=False)
        if resume_path else None
    )
    if resume_path and resume_ckpt:
        args.resume_from = resume_path
    if resume_ckpt:
        saved_cfg = resume_ckpt["config"]
        for key in ("d_model", "n_heads", "n_layers", "n_features", "block_size", "dropout",
                    "hex_influence", "tokenizer", "max_vocab_size", "bpe_vocab_size", "bpe_sample_chars"):
            if key in saved_cfg:
                setattr(args, key, saved_cfg[key])
    print(f"Using device: {device}")

    bulk_store = None
    if args.data_dir:
        saved_tokenizer = (
            BPETokenizer(merges=resume_ckpt["merges"])
            if resume_ckpt and resume_ckpt.get("merges") is not None
            else None
        )
        tok, bulk_store, bulk_metadata = build_or_load_bulk_tokens(
            args.data_dir, args.bulk_cache_dir,
            bpe_vocab_size=args.bpe_vocab_size,
            bpe_sample_chars=args.bpe_sample_chars,
            tokenizer=saved_tokenizer,
            rebuild=args.rebuild_bulk_cache,
        )
        # Keep the int32 memory-map view. Converting here would duplicate the
        # full cache as an 8-byte-per-token tensor in RAM.
        data = bulk_store.tensor
        text = None
        corpus_description = f"{bulk_metadata['file_count']:,} books"
    else:
        with open(args.data, "r", encoding="utf-8") as f:
            text = f.read()
        tok_cls = tokenizer_class(args.tokenizer)
        if resume_ckpt:
            if args.tokenizer == "bpe" and resume_ckpt.get("merges") is not None:
                tok = BPETokenizer(merges=resume_ckpt["merges"])
            else:
                tok = tok_cls(vocab=resume_ckpt["vocab"])
        elif args.tokenizer == "bpe":
            tok = BPETokenizer(text, vocab_size=args.bpe_vocab_size, sample_chars=args.bpe_sample_chars)
        elif args.tokenizer == "word":
            tok = WordTokenizer(text, max_vocab_size=args.max_vocab_size)
        else:
            tok = CharTokenizer(text)
        word_tokens = len(WordTokenizer._tokenize(text)) if args.tokenizer == "word" else 0
        warning = word_tokenizer_warning(args.tokenizer, word_tokens, tok.vocab_size)
        if warning:
            print(warning)
        if args.tokenizer == "bpe":
            data_path = os.path.abspath(args.data)
            cache_path = f"{data_path}.bpe{args.bpe_vocab_size}.encoded.pt"
            if os.path.exists(cache_path):
                data = torch.load(cache_path, map_location="cpu")
                print(f"Using cached BPE encoding: {cache_path}")
            else:
                data = torch.tensor(tok.encode(text), dtype=torch.long)
                torch.save(data, cache_path)
                print(f"Performed fresh BPE encode and cached it at: {cache_path}")
        else:
            data = torch.tensor(tok.encode(text), dtype=torch.long)
        corpus_description = f"{len(text):,} chars"
    if len(data) <= args.block_size + 1:
        raise ValueError("corpus must contain more tokens than block_size + 1")
    train_data, val_data, train_starts, val_starts = split_training_windows(
        data, args.block_size, strategy=args.validation_split, seed=args.validation_split_seed,
    )
    print(f"Corpus: {corpus_description}, {len(data):,} tokens, vocab size {tok.vocab_size} ({args.tokenizer})")

    auto_size_result = None
    if args.auto_size_model and not resume_ckpt:
        auto_size_result = auto_size_model(
            len(data), tok.vocab_size, args.block_size, args.n_heads,
            args.auto_size_target_tokens_per_param, args.min_tokens_per_param,
        )
        args.d_model = auto_size_result.d_model
        args.n_heads = auto_size_result.n_heads
        args.n_layers = auto_size_result.n_layers
        args.n_features = auto_size_result.n_features
        print("Auto-size selection: "
              f"corpus_tokens={len(data):,} | d_model={args.d_model} | n_heads={args.n_heads} "
              f"| n_layers={args.n_layers} | n_features={args.n_features} "
              f"| parameters={auto_size_result.parameter_count:,} "
              f"| tokens/parameter={auto_size_result.tokens_per_param:.3f}")
        if auto_size_result.status == "hard_floor":
            safe_parameter_budget = len(data) / args.min_tokens_per_param
            print("[WARNING] Even the minimum viable GamaX1 architecture "
                  f"(d_model={args.d_model}, n_layers={args.n_layers}, n_features={args.n_features}, "
                  f"~{auto_size_result.parameter_count:,} parameters) exceeds the recommended token budget "
                  f"for this corpus (~{len(data):,} tokens; {args.min_tokens_per_param:g} tokens/parameter "
                  f"minimum recommends staying under ~{safe_parameter_budget:,.0f} parameters). Proceeding "
                  "with the minimum viable architecture anyway, but expect some memorization risk with a corpus "
                  "this small -- consider adding more training data.")
        elif auto_size_result.status == "above_minimum_below_target":
            print("Auto-size status: above the minimum safety threshold but below the requested target; "
                  "manual validation is recommended.")
        else:
            print("Auto-size status: comfortably above the requested token-per-parameter target.")

    model = GamaX1Model(
        vocab_size=tok.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, n_features=args.n_features, max_seq_len=args.block_size,
        dropout=args.dropout,
        hex_influence=args.hex_influence, sparsity_k_init=max(1, args.n_features // 2),
        sparsity_k_min=max(1, args.n_features // 8),
    ).to(device)
    use_amp = device.startswith("cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("CUDA mixed precision: enabled (use --no_amp to disable)")
    optimizer = create_optimizer(model, args.lr, args.weight_decay)
    start_step = 0
    if resume_ckpt:
        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        model.sparsity_ctrl.load_state_dict(resume_ckpt["sparsity_controller_state"])
        if resume_ckpt.get("scaler_state"):
            scaler.load_state_dict(resume_ckpt["scaler_state"])
        if resume_ckpt.get("rng_state") is not None:
            torch.set_rng_state(resume_ckpt["rng_state"])
        if torch.cuda.is_available() and resume_ckpt.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_ckpt["cuda_rng_state_all"])
        start_step = int(resume_ckpt["step"])
        print(f"Resuming from step {start_step}: {args.resume_from}")
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    corpus_ratio = tokens_per_parameter(len(data), parameter_count)
    print(f"Model parameters: {parameter_count:,}")
    print(f"Corpus/model size check: ~{len(data):,} tokens, ~{parameter_count:,} parameters "
          f"(~{corpus_ratio:.3f} tokens/parameter). Recommended minimum is roughly "
          f"{args.min_tokens_per_param:g} tokens/parameter to reduce memorization risk.")

    os.makedirs(args.out_dir, exist_ok=True)
    val_history, train_history = [], []
    recent_training_losses = []
    memorization_warning_printed = False
    t0 = time.time()
    for step in range(start_step + 1, args.max_steps + 1):
        lr = get_lr_schedule(step, args.max_steps, args.lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device, train_starts)
        k = model.sparsity_ctrl.k
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, loss = model(xb, targets=yb, k=k)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        model.sparsity_ctrl.step(loss.item())
        recent_training_losses.append(loss.item())
        recent_training_losses = recent_training_losses[-args.eval_batches:]

        should_eval = step % args.eval_interval == 0 or step == start_step + 1
        overfit = False
        memorization = False
        if should_eval:
            model.eval()
            val_value = evaluate_loss(
                model, val_data, args.block_size, args.eval_batch_size,
                args.eval_batches, device, k=model.sparsity_ctrl.k, start_indices=val_starts,
                use_amp=use_amp,
            )
            train_value = sum(recent_training_losses) / len(recent_training_losses)
            train_history.append(train_value)
            val_history.append(val_value)
            val_history, train_history = val_history[-10:], train_history[-10:]
            active = model.active_units_per_token()
            ratio = args.n_features * args.n_layers / max(active, 1)
            print(f"step {step:5d} | lr {lr:.2e} | train_loss {train_value:.4f} | train_ppl {perplexity(train_value):.2f} "
                  f"| val_loss {val_value:.4f} | val_ppl {perplexity(val_value):.2f} "
                  f"| sparsity_k {k} | active_units/token {active} | compute_ratio_vs_dense {ratio:.2f}x "
                  f"| {time.time() - t0:.1f}s")
            overfit = is_overfitting(val_history, train_history, args.overfit_patience)
            if overfit:
                print(f"[WARNING] Validation loss has increased for {args.overfit_patience} consecutive evals while "
                      "training loss keeps dropping — this usually means the model is starting to memorize the "
                      "training data rather than generalize. Consider: a larger/more varied corpus, early stopping, "
                      "or reducing model size.")
            memorization = is_memorization_detected(
                len(data), parameter_count, train_value, val_value,
                args.min_tokens_per_param, args.perplexity_memorization_floor,
            )
            if memorization and not memorization_warning_printed:
                observed_ppl = min(perplexity(train_value), perplexity(val_value))
                print("[WARNING] MEMORIZATION DETECTED: perplexity is "
                      f"{observed_ppl:.2f}, close to the theoretical minimum of 1.0, and the corpus has only "
                      f"~{len(data):,} tokens against a ~{parameter_count:,}-parameter model "
                      f"(~{corpus_ratio:.3f} tokens per parameter, below the --min_tokens_per_param threshold "
                      f"of {args.min_tokens_per_param:g}). The model has likely memorized the training corpus "
                      "verbatim rather than learning generalizable language patterns. Fix by: (a) using a much "
                      "larger corpus, (b) reducing model size (fewer/smaller layers, --n_features), or (c) both.")
                memorization_warning_printed = True

        if args.checkpoint_interval and step % args.checkpoint_interval == 0:
            save_checkpoint(
                os.path.join(args.out_dir, f"gamax1_step_{step}.pt"),
                model, optimizer, tok, vars(args), step, scaler=scaler,
            )
            save_checkpoint(
                os.path.join(args.out_dir, "gamax1_latest.pt"),
                model, optimizer, tok, vars(args), step, scaler=scaler,
            )
        if (overfit or memorization) and args.early_stop_on_overfit:
            print("Early stopping because --early_stop_on_overfit was set.")
            break

    final_step = step if "step" in locals() else start_step
    ckpt_path = os.path.join(args.out_dir, "gamax1.pt")
    save_checkpoint(
        ckpt_path, model, optimizer, tok, vars(args), final_step, scaler=scaler
    )
    save_checkpoint(
        os.path.join(args.out_dir, "gamax1_latest.pt"),
        model, optimizer, tok, vars(args), final_step, scaler=scaler
    )
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))
    if bulk_store is not None:
        del data
        bulk_store.close()


if __name__ == "__main__":
    main()
