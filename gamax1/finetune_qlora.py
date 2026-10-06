"""
GamaX1 QLoRA-style instruction/Q&A fine-tuning.

Architecture-specific targets:
    blocks.*.attn.qkv
    blocks.*.attn.out_proj
    blocks.*.ffn.sparse.in_proj
    blocks.*.ffn.sparse.out_proj

Base model:
    4-bit NF4 quantized and frozen.

Trainable:
    LoRA A/B matrices only.

Loss:
    Answer-only masked cross entropy.

Sparsity:
    Frozen at the value stored in the base checkpoint.

Requires:
    pip install bitsandbytes
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import bitsandbytes as bnb
except ImportError as exc:
    raise RuntimeError(
        "bitsandbytes is required for 4-bit QLoRA training.\n"
        "Install it with:\n"
        "    pip install -U bitsandbytes"
    ) from exc

from .model import GamaX1Model
from .tokenizer import BPETokenizer
from .train import perplexity, get_lr_schedule
from .instruction_data import (
    load_pairs,
    InstructionDataset,
    make_collate_fn,
    split_train_val,
)


# ---------------------------------------------------------------------------
# LoRA + 4-bit Linear
# ---------------------------------------------------------------------------

class LoRA4bitLinear(nn.Module):
    """
    Frozen bitsandbytes 4-bit Linear + trainable LoRA adapters.

    y = W_4bit(x) + alpha/r * B(A(x))
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
        compute_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        self.base = bnb.nn.Linear4bit(
            self.in_features,
            self.out_features,
            bias=linear.bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=True,
            quant_type="nf4",
        )

        # Keep original weights on CPU until the model is moved to CUDA.
        #
        # bitsandbytes performs the actual 4-bit quantization when the
        # Linear4bit module is moved to CUDA.
        self.base.weight = bnb.nn.Params4bit(
            linear.weight.detach().cpu(),
            requires_grad=False,
            compress_statistics=True,
            quant_type="nf4",
        )

        if linear.bias is not None:
            self.base.bias = nn.Parameter(
                linear.bias.detach().cpu(),
                requires_grad=False,
            )

        # Explicitly freeze base.
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA parameters remain full precision.
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, dtype=torch.float32)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        # LoRA branch.
        #
        # Keep LoRA computation in fp32 for numerical stability and cast
        # the result back to the base output dtype.
        x_lora = self.dropout(x).to(self.lora_A.dtype)

        lora = F.linear(x_lora, self.lora_A)
        lora = F.linear(lora, self.lora_B)

        lora = lora * self.scaling
        lora = lora.to(base_out.dtype)

        return base_out + lora


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

TARGET_SUFFIXES = (
    ".attn.qkv",
    ".attn.out_proj",
    ".ffn.sparse.in_proj",
    ".ffn.sparse.out_proj",
)


def is_lora_target(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False

    return name.endswith(TARGET_SUFFIXES)


def replace_linear_modules(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    """
    Replace selected GamaX1 Linear modules with LoRA4bitLinear.
    """

    replacements = []

    # Collect first because the module tree is modified during traversal.
    candidates = []

    for name, module in model.named_modules():
        if is_lora_target(name, module):
            candidates.append((name, module))

    if not candidates:
        raise RuntimeError(
            "No GamaX1 LoRA target modules were found. "
            "Expected attention qkv/out_proj and sparse FFN projections."
        )

    for full_name, old_module in candidates:
        parts = full_name.split(".")
        parent = model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        child_name = parts[-1]

        new_module = LoRA4bitLinear(
            old_module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            compute_dtype=torch.float16,
        )

        setattr(parent, child_name, new_module)
        replacements.append(full_name)

    return replacements


def freeze_everything(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def trainable_parameters(model: nn.Module):
    return [
        p for p in model.parameters()
        if p.requires_grad
    ]


def count_parameters(parameters):
    return sum(p.numel() for p in parameters)


# ---------------------------------------------------------------------------
# Base checkpoint
# ---------------------------------------------------------------------------

def load_base_checkpoint(
    ckpt_path: str,
    device: str,
):
    """
    Load the original GamaX1 checkpoint on CPU.

    We intentionally keep the model in FP32 until the selected Linear
    modules have been converted to 4-bit.
    """

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = ckpt["config"]

    tok = BPETokenizer(
        merges=ckpt["merges"]
    )

    model = GamaX1Model(
        vocab_size=tok.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        n_features=cfg["n_features"],
        max_seq_len=cfg["block_size"],
        hex_influence=cfg.get("hex_influence", False),
        sparsity_k_init=max(
            1,
            cfg["n_features"] // 2,
        ),
        sparsity_k_min=max(
            1,
            cfg["n_features"] // 8,
        ),
        dense_mode=cfg.get("dense_mode", False),
    )

    model.load_state_dict(
        ckpt["model_state"],
        strict=True,
    )

    model.sparsity_ctrl.load_state_dict(
        ckpt["sparsity_controller_state"]
    )

    model.load_ptm_state_dicts(
        ckpt.get("ptm_states")
    )

    return model, tok, cfg


# ---------------------------------------------------------------------------
# Masked answer loss
# ---------------------------------------------------------------------------

def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute CE only on answer tokens.
    """

    flat_logits = logits.reshape(
        -1,
        logits.size(-1),
    )

    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)

    if not flat_mask.any():
        return logits.sum() * 0.0

    per_token = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="none",
    )

    return per_token[flat_mask].mean()


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    device,
    k,
    use_amp,
):
    model.eval()

    losses = []

    for xb, yb, mask in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits, _ = model(
                xb,
                targets=None,
                k=k,
                use_ptm=False,
            )

            loss = masked_cross_entropy(
                logits,
                yb,
                mask,
            )

        losses.append(loss.item())

    return sum(losses) / max(len(losses), 1)


# ---------------------------------------------------------------------------
# Adapter checkpoint
# ---------------------------------------------------------------------------

def collect_lora_state(model):
    """
    Save only trainable LoRA parameters.

    This keeps the fine-tune checkpoint small and independent from
    the frozen 4-bit base weights.
    """

    state = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRA4bitLinear):
            state[f"{name}.lora_A"] = (
                module.lora_A.detach().cpu()
            )
            state[f"{name}.lora_B"] = (
                module.lora_B.detach().cpu()
            )

    return state


def save_adapter_checkpoint(
    path,
    model,
    optimizer,
    tokenizer,
    base_checkpoint,
    base_config,
    step,
    best_val,
    args,
):
    os.makedirs(
        os.path.dirname(
            os.path.abspath(path)
        ),
        exist_ok=True,
    )

    payload = {
        "format": "gamax1_qlora_v1",

        "base_checkpoint": os.path.abspath(
            base_checkpoint
        ),

        "step": step,
        "best_val_loss": best_val,

        "lora": collect_lora_state(model),

        "lora_config": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": list(TARGET_SUFFIXES),
            "quant_type": "nf4",
        },

        "base_config": base_config,

        "sparsity_k": int(
            model.sparsity_ctrl.k
        ),

        "tokenizer_merges": tokenizer.merges,

        "optimizer_state": optimizer.state_dict(),

        "training_args": vars(args),
    }

    tmp = path + ".tmp"

    torch.save(
        payload,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )

    print(
        f"Saved adapter checkpoint to {path}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "GamaX1 4-bit NF4 + LoRA instruction/Q&A fine-tuning"
        )
    )

    parser.add_argument(
        "--init_from",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--data",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--out_dir",
        default="checkpoints_qlora",
        type=str,
    )

    parser.add_argument(
        "--max_len",
        default=512,
        type=int,
    )

    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--epochs",
        default=3,
        type=int,
    )

    parser.add_argument(
        "--lr",
        default=2e-4,
        type=float,
    )

    parser.add_argument(
        "--warmup_steps",
        default=50,
        type=int,
    )

    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--val_fraction",
        default=0.05,
        type=float,
    )

    parser.add_argument(
        "--eval_interval",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--checkpoint_interval",
        default=500,
        type=int,
    )

    parser.add_argument(
        "--lora_rank",
        default=16,
        type=int,
    )

    parser.add_argument(
        "--lora_alpha",
        default=32.0,
        type=float,
    )

    parser.add_argument(
        "--lora_dropout",
        default=0.05,
        type=float,
    )

    parser.add_argument(
        "--seed",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--no_amp",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = (
        args.device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    if not device.startswith("cuda"):
        raise RuntimeError(
            "4-bit bitsandbytes QLoRA requires CUDA "
            "for this implementation."
        )

    torch.manual_seed(args.seed)

    print("=" * 72)
    print("GamaX1 QLoRA fine-tuning")
    print("=" * 72)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # Load base
    # ---------------------------------------------------------

    model, tok, base_cfg = load_base_checkpoint(
        args.init_from,
        device="cpu",
    )

    if args.max_len > base_cfg["block_size"]:
        raise ValueError(
            f"--max_len={args.max_len} exceeds "
            f"base block_size={base_cfg['block_size']}"
        )

    print(
        f"Loaded base checkpoint: {args.init_from}"
    )

    print(
        f"Base step: {base_cfg.get('step', '?')}"
    )

    print(
        f"Base sparsity k: "
        f"{model.sparsity_ctrl.k}"
    )

    # ---------------------------------------------------------
    # Freeze original model
    # ---------------------------------------------------------

    freeze_everything(model)

    # ---------------------------------------------------------
    # Replace selected Linear modules
    # ---------------------------------------------------------

    print("\nReplacing GamaX1 target Linear modules...")

    replaced = replace_linear_modules(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    print(
        f"LoRA modules installed: {len(replaced)}"
    )

    for name in replaced:
        print(
            f"  + {name}"
        )

    # Only LoRA params train.
    trainable = trainable_parameters(model)

    if not trainable:
        raise RuntimeError(
            "No trainable LoRA parameters found."
        )

    trainable_count = count_parameters(
        trainable
    )

    total_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nTotal parameters: "
        f"{total_count:,}"
    )

    print(
        f"Trainable LoRA parameters: "
        f"{trainable_count:,}"
    )

    print(
        f"Trainable fraction: "
        f"{100.0 * trainable_count / total_count:.4f}%"
    )

    # ---------------------------------------------------------
    # Move quantized model to GPU
    # ---------------------------------------------------------

    print("\nMoving quantized base to CUDA...")

    model = model.to(device)

    # Sparsity must stay frozen.
    k = int(
        model.sparsity_ctrl.k
    )

    print(
        f"Frozen sparsity k = {k}"
    )

    # ---------------------------------------------------------
    # Instruction dataset
    # ---------------------------------------------------------

    pairs, stats = load_pairs(
        args.data
    )

    print(
        f"\nInstruction data:"
        f" {stats['pairs']} usable pairs"
    )

    if not pairs:
        raise ValueError(
            "No usable instruction/Q&A examples found."
        )

    train_pairs, val_pairs = split_train_val(
        pairs,
        args.val_fraction,
        seed=args.seed,
    )

    print(
        f"Train examples: {len(train_pairs)}"
    )

    print(
        f"Val examples: {len(val_pairs)}"
    )

    collate = make_collate_fn(
        tok.pad_id
    )

    train_loader = torch.utils.data.DataLoader(
        InstructionDataset(
            train_pairs,
            tok,
            args.max_len,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    val_loader = torch.utils.data.DataLoader(
        InstructionDataset(
            val_pairs,
            tok,
            args.max_len,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_amp = (
        device.startswith("cuda")
        and not args.no_amp
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    steps_per_epoch = max(
        1,
        len(train_loader),
    )

    max_steps = (
        steps_per_epoch * args.epochs
    )

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    best_val = float("inf")
    step = 0
    t0 = time.time()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print("\nStarting QLoRA training...")
    print(
        f"Max steps: {max_steps}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        for xb, yb, mask in train_loader:

            step += 1

            lr = get_lr_schedule(
                step,
                max_steps,
                args.lr,
                args.warmup_steps,
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            xb = xb.to(
                device,
                dtype=torch.long,
            )

            yb = yb.to(
                device,
                dtype=torch.long,
            )

            mask = mask.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits, _ = model(
                    xb,
                    targets=None,
                    k=k,
                    use_ptm=False,
                )

                loss = masked_cross_entropy(
                    logits,
                    yb,
                    mask,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                trainable,
                1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            if (
                step % args.eval_interval == 0
                or step == 1
            ):

                val_loss = evaluate(
                    model,
                    val_loader,
                    device,
                    k,
                    use_amp,
                )

                elapsed = (
                    time.time() - t0
                )

                print(
                    f"epoch {epoch} | "
                    f"step {step:5d}/{max_steps} | "
                    f"lr {lr:.2e} | "
                    f"train_loss {loss.item():.4f} | "
                    f"val_loss {val_loss:.4f} | "
                    f"val_ppl {perplexity(val_loss):.2f} | "
                    f"k {k} | "
                    f"{elapsed:.1f}s"
                )

                if val_loss < best_val:

                    best_val = val_loss

                    save_adapter_checkpoint(
                        os.path.join(
                            args.out_dir,
                            "gamax1_qlora_best.pt",
                        ),
                        model,
                        optimizer,
                        tok,
                        args.init_from,
                        base_cfg,
                        step,
                        best_val,
                        args,
                    )

            # -------------------------------------------------
            # Periodic checkpoint
            # -------------------------------------------------

            if (
                args.checkpoint_interval
                and step % args.checkpoint_interval == 0
            ):

                save_adapter_checkpoint(
                    os.path.join(
                        args.out_dir,
                        f"gamax1_qlora_step_{step}.pt",
                    ),
                    model,
                    optimizer,
                    tok,
                    args.init_from,
                    base_cfg,
                    step,
                    best_val,
                    args,
                )

    # ---------------------------------------------------------
    # Final checkpoint
    # ---------------------------------------------------------

    save_adapter_checkpoint(
        os.path.join(
            args.out_dir,
            "gamax1_qlora_latest.pt",
        ),
        model,
        optimizer,
        tok,
        args.init_from,
        base_cfg,
        step,
        best_val,
        args,
    )

    # Save tokenizer merges separately.
    with open(
        os.path.join(
            args.out_dir,
            "tokenizer_merges.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            tok.merges,
            f,
            ensure_ascii=False,
        )

    print("\n" + "=" * 72)
    print("QLoRA fine-tuning complete")
    print("=" * 72)

    print(
        f"Best validation loss: "
        f"{best_val:.4f}"
    )

    print(
        f"Best validation perplexity: "
        f"{perplexity(best_val):.2f}"
    )

    print(
        f"Final sparsity k: {k}"
    )

    print(
        f"Adapters saved under: "
        f"{args.out_dir}"
    )


if __name__ == "__main__":
    main()