"""Train matched sparse and dense GamaX1 variants for a small fair comparison."""

import argparse
import os

import torch

from .model import GamaX1Model
from .tokenizer import BPETokenizer, CharTokenizer, WordTokenizer
from .train import get_batch, perplexity


def evaluate(model, data, block_size, batch_size, device):
    model.eval()
    with torch.no_grad():
        xb, yb = get_batch(data, block_size, batch_size, device)
        _, loss = model(xb, targets=yb, k=model.sparsity_ctrl.k, use_ptm=False)
    return loss.item()


def train_pair(sparse, dense, train_data, val_data, args, device):
    """Train both models on identical sampled batches and optimizer settings."""
    sparse_opt = torch.optim.AdamW(sparse.parameters(), lr=args.lr)
    dense_opt = torch.optim.AdamW(dense.parameters(), lr=args.lr)
    for _ in range(args.steps):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        for model, optimizer in ((sparse, sparse_opt), (dense, dense_opt)):
            model.train()
            _, loss = model(xb, targets=yb, k=model.sparsity_ctrl.k)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if not model.dense_mode:
                model.sparsity_ctrl.step(loss.item())
    return {
        "sparse": (evaluate(sparse, train_data, args.block_size, args.batch_size, device),
                   evaluate(sparse, val_data, args.block_size, args.batch_size, device)),
        "dense": (evaluate(dense, train_data, args.block_size, args.batch_size, device),
                  evaluate(dense, val_data, args.block_size, args.batch_size, device)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare GamaX1 sparse FFN against a matched dense FFN.")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "sample_corpus.txt"))
    parser.add_argument("--tokenizer", choices=("char", "word", "bpe"), default="char")
    parser.add_argument("--bpe_vocab_size", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_features", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    if args.tokenizer == "bpe":
        tok = BPETokenizer(text, vocab_size=args.bpe_vocab_size, sample_chars=3_000_000)
    else:
        tok = (WordTokenizer if args.tokenizer == "word" else CharTokenizer)(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    common = dict(vocab_size=tok.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                  n_layers=args.n_layers, n_features=args.n_features, max_seq_len=args.block_size,
                  sparsity_k_init=args.n_features // 2, sparsity_k_min=args.n_features // 8)
    torch.manual_seed(0)
    sparse = GamaX1Model(**common).to(device)
    torch.manual_seed(0)
    dense = GamaX1Model(**common, dense_mode=True).to(device)
    results = train_pair(sparse, dense, train_data, val_data, args, device)

    sparse_compute = sparse.active_units_per_token()
    dense_compute = dense.active_units_per_token()
    ratio = dense_compute / max(sparse_compute, 1)
    sparse_val, dense_val = results["sparse"][1], results["dense"][1]
    retention = perplexity(dense_val) / max(perplexity(sparse_val), 1e-12) * 100
    print("\nFinal comparison")
    print("model  | train_loss | val_loss | val_ppl | parameters | active_units/token")
    for name, model in (("sparse", sparse), ("dense ", dense)):
        train_loss, val_loss = results[name.strip()]
        print(f"{name} | {train_loss:10.4f} | {val_loss:8.4f} | {perplexity(val_loss):7.2f} | "
              f"{sum(p.numel() for p in model.parameters()):10,d} | {model.active_units_per_token():18,d}")
    print(f"\nCompute ratio: {ratio:.2f}x (sparse uses {sparse_compute / dense_compute * 100:.1f}% of dense compute)")
    print(f"Accuracy/compute tradeoff: sparse retains {retention:.1f}% of dense's validation perplexity at "
          f"{sparse_compute / dense_compute * 100:.1f}% of the compute.")


if __name__ == "__main__":
    main()
