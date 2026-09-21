# GamaX1 — v2 changes (fresh codebase, better training approach)

This is a fresh drop of `tokenizer.py`, `bulk_corpus.py`, `train.py`,
`model.py`, `generate.py`, `layers.py`, `compare_dense.py`. Everything below
is tested (sandbox unit tests on synthetic data); none of it has been run on
your real corpus yet.

## 1. Four reserved special tokens, not just one

`tokenizer.py`'s `BPETokenizer` now reserves `<|eos|>`, `<|user|>`,
`<|assistant|>`, `<|pad|>` (in that order, right after the last BPE merge
id) instead of just `<|eos|>`. All derived from `len(self.merges)` — no
serialization change, an existing `tokenizer.json` still loads and gets all
four ids automatically.

- `tok.vocab_size` = `256 + merges + 4`
- `tok.eos_id`, `tok.user_id`, `tok.assistant_id`, `tok.pad_id`, or all of
  them at once via `tok.special_token_ids`
- `tok.decode(...)` no longer crashes on any of these ids
- `tok.decode_with_boundaries(...)` renders each one as a readable tag
  (`<|user|>`, `<|eos|>`, ...) instead of dropping it, so you can inspect a
  cache or a generation and see the actual structure
- `pad_id` is reserved but **not used anywhere yet** — bulk pretraining
  samples fixed-length windows from one continuous stream, so there's
  nothing to pad. It's there for a future instruction-tuning stage where
  individual examples have different lengths.

**Vocab-size arithmetic**: to land on an exact total (e.g. 16,000), pass
`--bpe_vocab_size <target - 4>` (e.g. `15996`), not `<target>`.

## 2. Per-source content format in `bulk_corpus.py` — the real fix for topic/format drift

The single biggest change. Previously every source got the same treatment:
whole file → tokens → one `eos_id` between files. Your own dataset-quality
report showed why that isn't enough:

- `books`/`books2`/`wiki` files start with Project Gutenberg's license
  preamble ("The Project Gutenberg eBook of... This eBook is for the use of
  anyone...") — repeated boilerplate across thousands of files, not content.
- `Discord-Dialogues` files glue several **unrelated** exchanges together
  with a standalone `---` line — a file-level-only boundary never
  separates these.
- `Reddit-Constructive` uses `Speaker 0:`/`Speaker 1:`, where the same
  label doesn't reliably mean the same role across threads — tagging
  these as `<|user|>`/`<|assistant|>` would be a fabricated, wrong signal.

Each source folder now has a configured **format** (`DEFAULT_SOURCE_FORMATS`
in `bulk_corpus.py`):

| Format | Sources (default) | What happens |
|---|---|---|
| `prose` | `books`, `books2`, `wiki` | Gutenberg license boilerplate stripped (same START/END markers as `prepare_large_corpus.py`); whole file is one `eos_id`-bounded unit |
| `user_assistant` | `Conversations-200k`, `Discord-Dialogues` | Split on a standalone `---` line first; within each resulting exchange, literal `User:`/`Assistant:` labels are replaced with the real `<|user|>`/`<|assistant|>` ids (not left as ordinary spoofable text); `eos_id` between every exchange, not just every file |
| `generic_turns` | `Reddit-Constructive` | Split on standalone `---` the same way, but **no role tags are fabricated** — each turn is still `eos_id`-bounded, just without a `<|user|>`/`<|assistant|>` guess |

A source not in this dict falls back to `prose` with a one-time warning
rather than silently guessing. Pass your own `source_formats={...}` dict to
`build_or_load_bulk_tokens()` to override without editing the file.

This is a **tokenizer-identity-relevant** change (it changes `vocab_size`)
and a content-encoding change, so it requires the same two things the
single-`eos_id` fix did: `--rebuild_bulk_cache` for a full re-encode, and a
fresh `--out_dir` (the vocab size changed again, from `+1` to `+4`).

## 3. `train.py`: the eval-batch-size OOM trap is fixed at the default level

`--eval_batch_size` no longer defaults to a fixed `128` regardless of your
training `--batch_size`. It now defaults to **whatever `--batch_size` is**,
resolved right after argument parsing. This was the direct cause of the
"step 1 succeeds, then CUDA OOM" crash you hit earlier (`batch_size=32`
training successfully, then `eval_batch_size=128` — 4x bigger — immediately
after, while training's memory was still reserved). Pass `--eval_batch_size`
explicitly if you've confirmed the GPU has headroom for a bigger eval batch.

## 4. `generate.py`: `--chat` mode actually uses the new role tokens

```bash
python -m gamax1.generate --ckpt checkpoints/gamax1_latest.pt --chat \
  --prompt "What is the capital of France?"
```

Wraps the prompt as `<|user|> ... <|assistant|>` using the real reserved
ids (not the literal text `"User:"`), then generates the reply. Implies
`--stop_at_eos` (add `--no_stop_at_eos` to override) so the assistant's
turn actually ends instead of running to `--max_new_tokens` regardless.
**Only meaningful on a checkpoint trained with the new `user_assistant`
source format** — on an older checkpoint the model never saw these ids and
`--chat` will error out rather than silently produce nonsense.

## 5. Everything from the previous EOS-fix round, carried forward and re-tested

- `model.py`: hierarchical-exit runs each block exactly once per token
  (incremental, not the old from-scratch-per-depth O(n²) recompute);
  `RouterExpert` honestly documented as constructed-but-unused/no-gradient.
- `generate.py`: restores the checkpoint's trained `sparsity_controller_state`
  (previously silently used the untrained initial `k`).
- `bulk_corpus.py`: file-identity check is size-only (not mtime), so a
  Google Drive remount no longer looks like "every file changed" and forces
  a false restart; `--rebuild_bulk_cache` quarantines old cache files with
  a `.bak` suffix instead of deleting them.
- Fixed a copy/paste mixup in this delivery where the top-level
  `compare_dense.py` wrapper and the real `gamax1/compare_dense.py`
  implementation (same filename, different content) had gotten swapped —
  both are now correct and import-tested.

## What this does NOT fix (still open, still recommended to defer)

- Router training (`Dynamic Resource Allocation` pillar) — needs a
  difficulty-labeling scheme, not done here.
- Curriculum-style training — no staged/difficulty-ordered schedule exists.
- Syneidesis (Embedded Safety) — separate, still-theoretical research paper.
- Arithmetic / precise factual recall — a corpus-format and tokenizer fix
  does not substitute for dedicated math data or more training+scale.
- A dedicated supervised instruction-tuning stage (masking the loss to
  only the assistant's tokens, rather than plain next-token prediction
  across the whole tagged stream) — the `<|user|>`/`<|assistant|>` tags
  make that possible to add later, but this delivery does not add
  answer-only loss masking to `train.py` itself.

## Required next steps to actually use this

1. Push these files to GitHub (same as before — Colab's `git clone`
   overwrites local runtime files every session).
2. `--rebuild_bulk_cache` for a full re-encode (vocab size changed again).
3. A fresh `--out_dir` / checkpoint directory (old checkpoints won't load —
   embedding/head width changed by 3 more rows).
4. Spot-check the new cache before spending hours training on it:
   ```python
   from gamax1.bulk_corpus import build_or_load_bulk_tokens
   tok, store, meta = build_or_load_bulk_tokens(DATA_DIR, BULK_CACHE_DIR, bpe_vocab_size=..., rebuild=True)
   print(tok.decode_with_boundaries(store.tensor[:2000].tolist()))
   ```
   and confirm the tags/boundaries look like what you expect for each
   source before committing to a full run.
