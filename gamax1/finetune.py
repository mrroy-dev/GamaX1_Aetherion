"""
gamax1/finetune.py
===================
Instruction/Q&A fine-tuning stage, run AFTER bulk pretraining (train.py)
on a base checkpoint. This is a deliberately separate script/loop from
train.py rather than a mode flag, because the two training regimes
differ in ways that shouldn't share code paths silently:

  - Data:   train.py samples random windows out of one continuous token
            stream (bulk_corpus.py). This script trains on discrete,
            padded (prompt, answer) examples (instruction_data.py).
  - Loss:   train.py computes loss on every token. This script computes
            loss ONLY on answer tokens (see instruction_data.py's
            loss_mask) -- so it does NOT use GamaX1Model.forward()'s
            built-in loss; it takes the logits and applies its own
            masked cross-entropy here.
  - LR/optimizer: a fresh, low-LR AdamW -- NOT the pretraining
            optimizer state (that state encodes a very different LR
            regime and would fight a low fine-tuning LR). Only the
            MODEL weights (and sparsity-controller/PTM state, so
            generation behavior stays consistent) are carried over.
  - Sparsity level: --freeze_sparsity (default True) keeps k fixed at
            whatever the base checkpoint converged to, rather than
            calling sparsity_ctrl.step() -- fine-tuning on a small
            dataset is exactly the situation the DynamicSparsityController
            was never validated for (its trend-window/patience logic
            assumes the long, noisy loss curve of bulk pretraining), so
            letting a short fine-tune run perturb k risked an
            uncontrolled, unvalidated side effect for no benefit.

OPTIONAL REPLAY (catastrophic-forgetting mitigation): pass
--replay_data_dir pointing at the same plain-.txt corpus used for
pretraining (or a subset of it) and --replay_ratio (e.g. 0.2) to mix
that fraction of steps as ordinary bulk-style next-token training,
interleaved with the instruction steps. This is optional and off by
default (ratio 0.0) -- turn it on if you observe the fine-tuned model's
general fluency degrading relative to the base checkpoint.
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

from .model import GamaX1Model
from .tokenizer import BPETokenizer
from .train import checkpoint_dict, perplexity, get_lr_schedule
from .instruction_data import (
    load_pairs, InstructionDataset, make_collate_fn, split_train_val,
)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged only over positions where mask is True.

    logits: (batch, seq, vocab); targets/mask: (batch, seq). If mask is
    all-False for a batch (shouldn't happen -- every example has at
    least one answer token -- but defensive against a pathological
    all-truncated batch), returns 0 with a warning rather than NaN from
    a zero-count division.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    if not flat_mask.any():
        print("[WARNING] a batch had zero loss-mask positions (fully truncated?); contributing 0 loss.")
        return logits.sum() * 0.0
    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return per_token[flat_mask].mean()


def load_base_checkpoint(ckpt_path: str, device: str):
    """Load a train.py checkpoint and rebuild the exact model/tokenizer it
    was saved with (same pattern as generate.py's load_model)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tok = BPETokenizer(merges=ckpt["merges"])
    model = GamaX1Model(
        vocab_size=tok.vocab_size,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        n_features=cfg["n_features"], max_seq_len=cfg["block_size"],
        hex_influence=cfg.get("hex_influence", False),
        sparsity_k_init=max(1, cfg["n_features"] // 2),
        sparsity_k_min=max(1, cfg["n_features"] // 8),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.sparsity_ctrl.load_state_dict(ckpt["sparsity_controller_state"])
    model.load_ptm_state_dicts(ckpt.get("ptm_states"))
    return model, tok, cfg


@torch.no_grad()
def evaluate(model, val_loader, device, k, use_amp):
    """Evaluate loss weighted by the number of answer/EOS tokens.

    Averaging batch means equally can bias validation when the final batch
    or examples have different numbers of answer tokens.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for xb, yb, mask in val_loader:
        xb, yb, mask = xb.to(device), yb.to(device), mask.to(device).bool()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits, _ = model(xb, targets=None, k=k, use_ptm=False)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = yb.reshape(-1)
            flat_mask = mask.reshape(-1)
            n_tokens = int(flat_mask.sum().item())
            if n_tokens == 0:
                continue
            per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            total_loss += float(per_token[flat_mask].sum().item())
            total_tokens += n_tokens
    if total_tokens == 0:
        raise ValueError("Validation split contains zero answer-token targets; cannot evaluate.")
    return total_loss / total_tokens


def save_finetune_checkpoint(path, model, optimizer, tok, config, step, scaler,
                             *, epoch, batch_in_epoch, best_val, max_steps, args):
    """Atomically save model plus optimizer, RNG, and exact resume position."""
    payload = checkpoint_dict(model, optimizer, tok, config, step, scaler=scaler)
    payload["fine_tune_state"] = {
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "best_val_loss": None if best_val == float("inf") else float(best_val),
        "max_steps": int(max_steps),
        "args": vars(args).copy(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    print(f"Saved checkpoint to {path} (step={step}, epoch={epoch}, batch={batch_in_epoch})")


def restore_rng_state(ckpt):
    """Restore RNG after setup so resumed training uses saved randomness."""
    if ckpt.get("rng_state") is not None:
        torch.set_rng_state(ckpt["rng_state"].cpu())
    cuda_states = ckpt.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_states is not None:
        try:
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception as exc:
            print(f"[WARNING] Could not fully restore CUDA RNG state: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained GamaX1 checkpoint on instruction/Q&A JSON data "
                     "with answer-only masked loss."
    )
    parser.add_argument("--init_from", type=str, default=None,
                         help="Pretrained checkpoint for a NEW fine-tuning run. Not needed with --resume/--resume_from.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true",
                              help="Resume from OUT_DIR/gamax1_finetune_latest.pt.")
    resume_group.add_argument("--resume_from", type=str, default=None,
                              help="Resume from a specific fine-tuning checkpoint (.pt).")
    parser.add_argument("--data", type=str, default=None,
                         help="Path to a .json/.jsonl file, or a directory of them (recursive). "
                              "See instruction_data.py for supported record formats.")
    parser.add_argument("--out_dir", type=str, default="checkpoints_finetune")
    parser.add_argument("--max_len", type=int, default=None,
                         help="Max tokens per example (prompt+answer+tags). Must not exceed "
                              "the base checkpoint's block_size.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                         help="Fine-tuning LR. Deliberately far below pretraining LR (often "
                              "1e-4 to 3e-4) so the model adjusts to answer facts/format "
                              "without unlearning general fluency.")
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--val_fraction", type=float, default=None)
    parser.add_argument("--eval_interval", type=int, default=None,
                         help="Evaluate on the held-out split every N steps.")
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--freeze_sparsity", action="store_true", default=None,
                         help="Keep sparsity_k fixed at the base checkpoint's converged value "
                              "instead of letting DynamicSparsityController adapt further "
                              "during fine-tuning (default: on -- see module docstring).")
    parser.add_argument("--no_freeze_sparsity", dest="freeze_sparsity", action="store_false")
    parser.set_defaults(freeze_sparsity=None)
    parser.add_argument("--replay_data_dir", type=str, default=None,
                         help="Optional: a plain-.txt bulk corpus directory to interleave as "
                              "ordinary next-token training, mitigating catastrophic "
                              "forgetting of general fluency. Off by default.")
    parser.add_argument("--replay_ratio", type=float, default=None,
                         help="Fraction of steps drawn from --replay_data_dir instead of the "
                              "instruction data (0.0-1.0). Only used if --replay_data_dir is set.")
    parser.add_argument("--replay_cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true", default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    resume_path = None
    if args.resume:
        resume_path = os.path.join(args.out_dir, "gamax1_finetune_latest.pt")
    elif args.resume_from:
        resume_path = args.resume_from

    if resume_path is None and not args.init_from:
        parser.error("For a new run, --init_from is required. Use --resume or --resume_from to continue.")
    if resume_path is not None and not os.path.isfile(resume_path):
        parser.error(f"Resume checkpoint does not exist: {resume_path}")

    # Seed defensively before model construction. CLI defaults are resolved
    # below (and a resume may inherit the seed saved in its checkpoint), so
    # args.seed can still be None at this point.
    initial_seed = 0 if args.seed is None else int(args.seed)
    torch.manual_seed(initial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initial_seed)

    source_checkpoint = resume_path if resume_path is not None else args.init_from
    model, tok, base_cfg = load_base_checkpoint(source_checkpoint, device)
    resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False) if resume_path else None

    # On resume, reuse saved training options unless the user explicitly
    # overrides them on the command line. This keeps the LR schedule, split,
    # batch size, and sparsity/replay choices consistent across sessions.
    saved_ft = {}
    if resume_ckpt is not None:
        saved_ft = (resume_ckpt.get("fine_tune_state") or {}).get("args") or {}
        if not saved_ft:
            saved_ft = base_cfg.get("fine_tuning", {}) or {}
    defaults = {
        "data": None, "max_len": 512, "batch_size": 8, "epochs": 3,
        "lr": 3e-5, "warmup_steps": 50, "weight_decay": 0.0,
        "val_fraction": 0.05, "eval_interval": 200, "checkpoint_interval": 500,
        "freeze_sparsity": True, "replay_data_dir": None, "replay_ratio": 0.0,
        "replay_cache_dir": "data/finetune_replay_cache", "seed": 0, "no_amp": False,
    }
    for name, fallback in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, saved_ft.get(name, fallback))
    if args.data is None:
        parser.error("--data is required for a new run; for resume, pass it or use a checkpoint containing saved fine-tuning metadata.")
    if not os.path.exists(args.data):
        parser.error(f"Training data path does not exist: {args.data}")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.max_len < 2:
        parser.error("--max_len must be >= 2")
    if args.lr <= 0:
        parser.error("--lr must be > 0")
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("--val_fraction must be in [0, 1)")
    if not 0.0 <= args.replay_ratio <= 1.0:
        parser.error("--replay_ratio must be in [0, 1]")
    if args.replay_ratio > 0 and not args.replay_data_dir:
        print("[WARNING] --replay_ratio was set but --replay_data_dir is missing; replay will be disabled.")
    if args.max_len > base_cfg["block_size"]:
        raise ValueError(f"--max_len={args.max_len} exceeds base checkpoint's block_size="
                          f"{base_cfg['block_size']}; the model was never trained on sequences "
                          "this long.")

    # Apply the resolved seed before building/splitting the dataset. A full
    # resume restores the saved RNG state after setup, immediately before train.
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    print(f"Loaded checkpoint from {source_checkpoint} (step {resume_ckpt.get('step', '?') if resume_ckpt else base_cfg.get('step', '?')})")
    print(f"Sparsity k: {model.sparsity_ctrl.k} (frozen: {args.freeze_sparsity})")

    pairs, stats = load_pairs(args.data)
    print(f"Instruction data: {stats['files']} file(s), {stats['records']} record(s) -> "
          f"{stats['pairs']} (prompt, answer) example(s); {stats['skipped_records']} record(s) "
          "skipped (unrecognized format)")
    if stats["skipped_records"] and stats["unmatched_key_sets"]:
        print("[WARNING] Sample unmatched record key sets (first few) -- if this is your real "
              "data format, tell me these keys and I'll add support for them:")
        for keys in stats["unmatched_key_sets"]:
            print(f"    {list(keys)}")
    if not pairs:
        raise ValueError(
            "No usable (prompt, answer) examples found in --data. See the format-detection "
            "rules documented at the top of instruction_data.py, or the skipped-record key "
            "sets printed above, and either reshape the data or tell me the actual key names."
        )

    train_pairs, val_pairs = split_train_val(pairs, args.val_fraction, seed=args.seed)
    print(f"Train examples: {len(train_pairs)} | Val examples: {len(val_pairs)}")

    collate = make_collate_fn(tok.pad_id)
    train_dataset = InstructionDataset(train_pairs, tok, args.max_len)
    val_loader = torch.utils.data.DataLoader(
        InstructionDataset(val_pairs, tok, args.max_len),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    )

    # Use a local generator per epoch instead of DataLoader(shuffle=True).
    # This makes each epoch's sample order reconstructible after a restart.
    def make_epoch_loader(epoch_number):
        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(args.seed + int(epoch_number))
        order = torch.randperm(len(train_dataset), generator=epoch_generator).tolist()
        sampler = order
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            shuffle=False, collate_fn=collate, generator=epoch_generator,
        )

    use_replay = args.replay_data_dir is not None and args.replay_ratio > 0.0
    replay_data = replay_starts = None
    if use_replay:
        # Reuses the exact same bulk-corpus token cache machinery train.py
        # uses, so replay batches are drawn from real pretraining-format
        # text (see bulk_corpus.py) with no separate code path to maintain.
        from .bulk_corpus import build_or_load_bulk_tokens
        from .train import get_batch
        _replay_tok, _replay_store, _sources = build_or_load_bulk_tokens(
            args.replay_data_dir, args.replay_cache_dir,
            tokenizer=tok, rebuild=False,
        )
        replay_data = _replay_store.tensor
        replay_starts = None   # None -> get_batch samples fully random windows
        print(f"Replay corpus: {len(replay_data):,} tokens from {args.replay_data_dir} "
              f"(mixed in at ratio {args.replay_ratio:g})")

    use_amp = device.startswith("cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, (len(train_dataset) + args.batch_size - 1) // args.batch_size)
    max_steps = steps_per_epoch * args.epochs
    os.makedirs(args.out_dir, exist_ok=True)

    step = 0
    t0 = time.time()
    best_val = float("inf")
    k = model.sparsity_ctrl.k
    start_epoch = 1
    start_batch = 0

    if resume_ckpt is not None:
        # Restore optimizer/scaler and exact position for checkpoints written by
        # this version. Older checkpoints lack batch position; support them with
        # an explicit warning and best-effort position inference.
        if resume_ckpt.get("optimizer_state") is None:
            raise ValueError("This checkpoint has no optimizer_state; it cannot resume training.")
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        if resume_ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state"])
        step = int(resume_ckpt.get("step", 0))
        state = resume_ckpt.get("fine_tune_state")
        if state is not None:
            start_epoch = max(1, int(state.get("epoch", 1)))
            start_batch = max(0, int(state.get("batch_in_epoch", 0)))
            saved_best = state.get("best_val_loss")
            if saved_best is not None:
                best_val = float(saved_best)
            elif base_cfg.get("fine_tuning", {}).get("best_val_loss") is not None:
                best_val = float(base_cfg["fine_tuning"]["best_val_loss"])
        else:
            if base_cfg.get("fine_tuning", {}).get("best_val_loss") is not None:
                best_val = float(base_cfg["fine_tuning"]["best_val_loss"])
            start_epoch = min(args.epochs + 1, step // steps_per_epoch + 1)
            start_batch = step % steps_per_epoch
            print("[WARNING] Legacy checkpoint has no epoch/batch metadata. Restoring model/optimizer, "
                  "but resume position and sample order are approximate; the next checkpoint will "
                  "record exact position.")
        print(f"Resuming from {resume_path}: step={step}, epoch={start_epoch}, "
              f"batch={start_batch}, best_val={best_val if best_val != float('inf') else 'unknown'}")
    else:
        print(f"Starting new fine-tuning run from {args.init_from}")

    # Metadata stored inside config is useful even in viewers that don't inspect
    # the top-level fine_tune_state field.
    ft_cfg = dict(base_cfg)
    ft_cfg["out_dir"] = os.path.abspath(args.out_dir)
    ft_cfg["run_name"] = os.path.basename(os.path.abspath(args.out_dir))
    ft_cfg["fine_tuning"] = vars(args).copy()
    ft_cfg["fine_tuning"]["best_val_loss"] = None if best_val == float("inf") else best_val

    if resume_ckpt is not None:
        # Restore RNG last: constructing loaders/optimizer must not consume the
        # saved sequence that the next training step expects.
        restore_rng_state(resume_ckpt)

    last_epoch = start_epoch
    last_batch_in_epoch = start_batch
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loader = make_epoch_loader(epoch)
        last_epoch = epoch
        for batch_idx, (xb, yb, mask) in enumerate(epoch_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            step += 1
            batch_in_epoch = batch_idx + 1
            lr = get_lr_schedule(step, max_steps, args.lr, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            model.train()
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device).bool()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, _ = model(xb, targets=None, k=k, use_ptm=not args.freeze_sparsity)
                loss = masked_cross_entropy(logits, yb, mask)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if not args.freeze_sparsity:
                model.sparsity_ctrl.step(loss.item())
                k = model.sparsity_ctrl.k

            if use_replay and torch.rand(1).item() < args.replay_ratio:
                xb_r, yb_r = get_batch(replay_data, args.max_len, args.batch_size, device, replay_starts)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    _, replay_loss = model(xb_r, targets=yb_r, k=k)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(replay_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            last_batch_in_epoch = batch_in_epoch
            if step % args.eval_interval == 0 or step == 1:
                val_loss = evaluate(model, val_loader, device, k, use_amp)
                print(f"epoch {epoch} | step {step:5d}/{max_steps} | lr {lr:.2e} | "
                      f"train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | "
                      f"val_ppl {perplexity(val_loss):.2f} | {time.time() - t0:.1f}s")
                if val_loss < best_val:
                    best_val = val_loss
                    ft_cfg["fine_tuning"]["best_val_loss"] = best_val
                    save_finetune_checkpoint(
                        os.path.join(args.out_dir, "gamax1_finetune_best.pt"),
                        model, optimizer, tok, ft_cfg, step, scaler,
                        epoch=epoch, batch_in_epoch=batch_in_epoch, best_val=best_val,
                        max_steps=max_steps, args=args,
                    )

            if args.checkpoint_interval and step % args.checkpoint_interval == 0:
                ft_cfg["fine_tuning"]["best_val_loss"] = None if best_val == float("inf") else best_val
                save_finetune_checkpoint(
                    os.path.join(args.out_dir, f"gamax1_finetune_step_{step}.pt"),
                    model, optimizer, tok, ft_cfg, step, scaler,
                    epoch=epoch, batch_in_epoch=batch_in_epoch, best_val=best_val,
                    max_steps=max_steps, args=args,
                )

    # If resuming from an already-completed checkpoint with the same --epochs,
    # no batch may run. Still produce a valid latest checkpoint and tokenizer.
    if last_epoch > args.epochs:
        last_epoch = args.epochs
        last_batch_in_epoch = steps_per_epoch
    ft_cfg["fine_tuning"]["best_val_loss"] = None if best_val == float("inf") else best_val
    save_finetune_checkpoint(
        os.path.join(args.out_dir, "gamax1_finetune_latest.pt"),
        model, optimizer, tok, ft_cfg, step, scaler,
        epoch=last_epoch, batch_in_epoch=last_batch_in_epoch, best_val=best_val,
        max_steps=max_steps, args=args,
    )
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))
    if best_val == float("inf"):
        print(f"\nDone. No validation score was recorded. Checkpoints saved under {args.out_dir}/")
    else:
        print(f"\nDone. Best val_loss: {best_val:.4f} (val_ppl {perplexity(best_val):.2f}). "
              f"Checkpoints saved under {args.out_dir}/")
    print("Note: fine-tuning improves behavior only to the extent supported by the data; it cannot guarantee zero hallucinations.")


if __name__ == "__main__":
    main()