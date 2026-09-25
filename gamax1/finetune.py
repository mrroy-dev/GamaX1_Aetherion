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
from .train import checkpoint_dict, save_checkpoint, perplexity, get_lr_schedule
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
    model.eval()
    losses = []
    for xb, yb, mask in val_loader:
        xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits, _ = model(xb, targets=None, k=k, use_ptm=False)
            loss = masked_cross_entropy(logits, yb, mask)
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained GamaX1 checkpoint on instruction/Q&A JSON data "
                     "with answer-only masked loss."
    )
    parser.add_argument("--init_from", type=str, required=True,
                         help="Path to the pretrained checkpoint to fine-tune (e.g. "
                              "gamax1_step_20000.pt from train.py).")
    parser.add_argument("--data", type=str, required=True,
                         help="Path to a .json/.jsonl file, or a directory of them (recursive). "
                              "See instruction_data.py for supported record formats.")
    parser.add_argument("--out_dir", type=str, default="checkpoints_finetune")
    parser.add_argument("--max_len", type=int, default=512,
                         help="Max tokens per example (prompt+answer+tags). Must not exceed "
                              "the base checkpoint's block_size.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5,
                         help="Fine-tuning LR. Deliberately far below pretraining LR (often "
                              "1e-4 to 3e-4) so the model adjusts to answer facts/format "
                              "without unlearning general fluency.")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--eval_interval", type=int, default=200,
                         help="Evaluate on the held-out split every N steps.")
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--freeze_sparsity", action="store_true", default=True,
                         help="Keep sparsity_k fixed at the base checkpoint's converged value "
                              "instead of letting DynamicSparsityController adapt further "
                              "during fine-tuning (default: on -- see module docstring).")
    parser.add_argument("--no_freeze_sparsity", dest="freeze_sparsity", action="store_false")
    parser.add_argument("--replay_data_dir", type=str, default=None,
                         help="Optional: a plain-.txt bulk corpus directory to interleave as "
                              "ordinary next-token training, mitigating catastrophic "
                              "forgetting of general fluency. Off by default.")
    parser.add_argument("--replay_ratio", type=float, default=0.0,
                         help="Fraction of steps drawn from --replay_data_dir instead of the "
                              "instruction data (0.0-1.0). Only used if --replay_data_dir is set.")
    parser.add_argument("--replay_cache_dir", type=str, default="data/finetune_replay_cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model, tok, base_cfg = load_base_checkpoint(args.init_from, device)
    if args.max_len > base_cfg["block_size"]:
        raise ValueError(f"--max_len={args.max_len} exceeds base checkpoint's block_size="
                          f"{base_cfg['block_size']}; the model was never trained on sequences "
                          "this long.")

    print(f"Loaded base checkpoint from {args.init_from} (step {base_cfg.get('step', '?')})")
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
    train_loader = torch.utils.data.DataLoader(
        InstructionDataset(train_pairs, tok, args.max_len),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        InstructionDataset(val_pairs, tok, args.max_len),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    )

    use_replay = args.replay_data_dir is not None and args.replay_ratio > 0.0
    replay_data = replay_starts = None
    if use_replay:
        # Reuses the exact same bulk-corpus token cache machinery train.py
        # uses, so replay batches are drawn from real pretraining-format
        # text (see bulk_corpus.py) with no separate code path to maintain.
        from .bulk_corpus import build_or_load_bulk_tokens
        from .train import get_batch
        replay_data, replay_starts, _sources = build_or_load_bulk_tokens(
            args.replay_data_dir, args.replay_cache_dir, tok, rebuild=False,
        )
        print(f"Replay corpus: {len(replay_data):,} tokens from {args.replay_data_dir} "
              f"(mixed in at ratio {args.replay_ratio:g})")

    use_amp = device.startswith("cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    max_steps = steps_per_epoch * args.epochs
    os.makedirs(args.out_dir, exist_ok=True)

    step = 0
    t0 = time.time()
    best_val = float("inf")
    k = model.sparsity_ctrl.k

    for epoch in range(1, args.epochs + 1):
        for xb, yb, mask in train_loader:
            step += 1
            lr = get_lr_schedule(step, max_steps, args.lr, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            model.train()
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)

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

            if step % args.eval_interval == 0 or step == 1:
                val_loss = evaluate(model, val_loader, device, k, use_amp)
                print(f"epoch {epoch} | step {step:5d}/{max_steps} | lr {lr:.2e} | "
                      f"train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | "
                      f"val_ppl {perplexity(val_loss):.2f} | {time.time() - t0:.1f}s")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        os.path.join(args.out_dir, "gamax1_finetune_best.pt"),
                        model, optimizer, tok, base_cfg, step, scaler=scaler,
                    )

            if args.checkpoint_interval and step % args.checkpoint_interval == 0:
                save_checkpoint(
                    os.path.join(args.out_dir, f"gamax1_finetune_step_{step}.pt"),
                    model, optimizer, tok, base_cfg, step, scaler=scaler,
                )

    save_checkpoint(
        os.path.join(args.out_dir, "gamax1_finetune_latest.pt"),
        model, optimizer, tok, base_cfg, step, scaler=scaler,
    )
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))
    print(f"\nDone. Best val_loss: {best_val:.4f} (val_ppl {perplexity(best_val):.2f}). "
          f"Checkpoints saved under {args.out_dir}/")


if __name__ == "__main__":
    main()
