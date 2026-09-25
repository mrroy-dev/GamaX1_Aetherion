"""
gamax1/generate.py
====================
Load a trained GamaX1 checkpoint and generate text.
"""

import argparse
import os

import torch

from .model import GamaX1Model
from .tokenizer import BPETokenizer, CharTokenizer, WordTokenizer


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("tokenizer") == "bpe":
        tok = BPETokenizer(merges=ckpt["merges"])
    else:
        tok_cls = WordTokenizer if cfg.get("tokenizer") == "word" else CharTokenizer
        tok = tok_cls(vocab=ckpt["vocab"])
    model = GamaX1Model(
        vocab_size=tok.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        n_features=cfg["n_features"],
        max_seq_len=cfg["block_size"],
        hex_influence=cfg.get("hex_influence", False),
        sparsity_k_init=max(1, cfg["n_features"] // 2),
        sparsity_k_min=max(1, cfg["n_features"] // 8),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    # Restore the trained sparsity level. DynamicSparsityController is a
    # plain Python object (not an nn.Module), so its state is NOT captured
    # by model.state_dict()/load_state_dict() above -- it's saved/restored
    # separately, exactly like train.py already does on resume. Without
    # this, generation would silently use the INITIAL k (n_features // 2)
    # instead of whatever k the controller actually converged to during
    # training, which can meaningfully change generation behavior since the
    # sparse layer's active-feature count would no longer match what the
    # model was trained and evaluated under.
    model.sparsity_ctrl.load_state_dict(ckpt["sparsity_controller_state"])
    model.eval()
    return model, tok


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained GamaX1 model.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/gamax1.pt")
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Penalize reusing tokens already in the sequence. >1.0 suppresses "
                             "repetition (e.g. 1.2); 1.0 disables it (default: 1.0).")
    parser.add_argument("--hierarchical_exit", action="store_true",
                         help="Use Router/Validator hierarchical early exit at inference time.")
    parser.add_argument("--stop_at_eos", action="store_true",
                         help="Stop generation as soon as the tokenizer's reserved "
                              "document-boundary token (eos_id) is produced, instead of "
                              "always generating max_new_tokens. Only meaningful for a "
                              "BPE tokenizer trained with the eos_id document-boundary fix "
                              "(bulk_corpus.py inserting eos_id between files instead of "
                              "\"\\n\\n\"); has no effect otherwise since the model will "
                              "essentially never produce that id.")
    parser.add_argument("--chat", action="store_true",
                         help="Wrap --prompt as a user turn using the tokenizer's reserved "
                              "<|user|>/<|assistant|> ids (instead of the literal text "
                              "\"User:\"/\"Assistant:\") and generate the assistant's reply. "
                              "Only meaningful for a checkpoint trained on the role-tagged "
                              "corpus format (bulk_corpus.py's user_assistant source format); "
                              "on any other checkpoint the model was never shown these ids "
                              "and this will not produce a sensible reply. Use "
                              "--stop_at_eos explicitly when the training data contains EOS turn boundaries.")
    parser.add_argument("--no_stop_at_eos", action="store_true",
                         help="With --chat, generate the full --max_new_tokens instead of "
                              "stopping at the assistant turn's eos_id.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be > 0")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if args.top_k < 0:
        parser.error("--top_k must be >= 0")
    if args.repetition_penalty <= 0:
        parser.error("--repetition_penalty must be > 0")
    if args.no_stop_at_eos and not args.chat and not args.stop_at_eos:
        parser.error("--no_stop_at_eos is only meaningful with --chat")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.ckpt, device)

    stop_at_eos = args.stop_at_eos and not args.no_stop_at_eos
    eos_id = tok.eos_id if (stop_at_eos and hasattr(tok, "eos_id")) else None

    if args.chat:
        if not hasattr(tok, "user_id") or not hasattr(tok, "assistant_id"):
            parser.error(
                "--chat requires a tokenizer with reserved <|user|>/<|assistant|> ids "
                "(this checkpoint's tokenizer does not have them -- it predates the "
                "role-tagged corpus format, or was trained without it)."
            )
        prompt_ids = [tok.user_id] + tok.encode(args.prompt) + [tok.assistant_id]
    else:
        prompt_ids = tok.encode(args.prompt)

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, repetition_penalty=args.repetition_penalty,
        use_hierarchical_exit=args.hierarchical_exit, eos_id=eos_id,
    )
    if args.chat:
        # Only the newly generated continuation is the assistant's reply;
        # decode_with_boundaries makes the prompt's own role tags visible
        # too, for inspection.
        print(tok.decode_with_boundaries(out[0].tolist()))
    else:
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()