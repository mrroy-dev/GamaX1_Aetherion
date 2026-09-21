# Bug fixes (from bug.md review)

All 5 reported issues verified against the code first; 4 were confirmed real
and fixed below (with a sandbox reproduction + regression test for each). One
was confirmed real but is a different kind of fix than requested.

## 1. `get_batch` off-by-one — CONFIRMED, FIXED
`train.py`: `torch.randint(len(data) - block_size - 1, ...)` →
`torch.randint(len(data) - block_size, ...)`. Reproduced the exact crash on
`len(data) == block_size + 1` before the fix; confirmed both the crash is
gone and the true last valid window (`len(data) - block_size - 1`) is now
reachable, without ever sampling past it, after the fix.

## 2. Repetition penalty used only batch row 0 — CONFIRMED, FIXED
`model.py`: `apply_repetition_penalty` now takes a `(batch, vocab)` mask
(built via `present.scatter_(1, idx, True)` in `generate()`) and applies it
elementwise with `torch.where`, instead of a single `(vocab,)` mask built
from `idx[0]` and applied to every row via boolean fancy-indexing (which
also collapsed the batch dimension). Verified per-row correctness directly.
Note: `generate.py`'s CLI always calls with batch=1, so this bug was latent
(no visible effect) in current usage — it matters for anyone calling
`model.generate()` with batch>1 directly.

## 3. PTM nudge value was detached, cutting its own gradient — CONFIRMED, FIXED
`layers.py`, `SparseSuperpositionLinear.forward`: removed `.detach()` from
`pre[..., nudge_indices].detach() * 0.5 + 1e-3`. Verified directly that the
nudge value's `requires_grad`/`grad_fn` go from `False`/`None` (broken) to
`True`/present (fixed) — confirms the report's claim that the previous code
contradicted its own comment ("keeps receiving gradient").

## 4. Checkpoint resume lost PTM state and sparsity-trend history — CONFIRMED, FIXED
- `layers.py`: `DynamicSparsityController.state_dict()`/`load_state_dict()`
  now include `loss_history` and `steps_since_change` (previously only `k`
  and `exploration_fraction`). Backward-compatible: `.get()` with a safe
  default so an older checkpoint without these keys still loads.
- `layers.py`: `ProbationaryMemoryTracker` gained its own
  `state_dict()`/`load_state_dict()` (it's a plain object, not an
  `nn.Module`, so it was never covered by `model.state_dict()` at all).
- `model.py`: `GamaX1Model.ptm_state_dicts()` / `.load_ptm_state_dicts()`
  collect/restore every block's PTM in one call.
- `train.py`: checkpoint save now includes `"ptm_states"`; resume calls
  `model.load_ptm_state_dicts(resume_ckpt.get("ptm_states"))`
  (`.get()` — tolerant of an older checkpoint that predates this fix).
- Verified round-trip at both the standalone-object level and the
  full-`GamaX1Model` level.

## 5. Curated Gutenberg list was dead code — CONFIRMED, DIFFERENT FIX APPLIED
`prepare_large_corpus.py`: `DEFAULT_BOOK_IDS` was assigned the curated list,
then immediately overwritten with `list(range(1, 401))` a few lines later —
so the curated list was unreachable code, and the plain range silently ran
whenever `--ids`/`--gutenberg_id` weren't given.

Rather than just deleting the dead assignment, both lists are now real,
independently named presets (`CURATED_BOOK_IDS`, `RANGE_BOOK_IDS`), selectable
via a new `--preset {curated,range_1_400}` flag. Default is `range_1_400` —
matching what was actually running before this fix (not the curated list),
so this does not silently change anyone's existing default behavior; it just
makes the choice visible and lets you opt into the curated list explicitly.
Running with no `--ids`/`--preset` now also prints which preset it picked.

## Verification performed here
- `ast.parse()` on every changed file (syntax).
- Full package import (`from gamax1 import ...`).
- A sandbox reproduction test for each of the 4 code bugs, both before and
  after the fix (confirming the failure mode existed, then confirming it's
  gone) -- not just "does it run", the actual claimed defect.
- A regression pass re-confirming the earlier fixes from this conversation
  (hierarchical-exit O(n) incremental depth, eos_id early-stop, per-source
  bulk_corpus.py encoding) still work correctly with these changes layered
  on top.

## Not done here (per bug.md's own "Definition of done", still open)
- New unit tests added to a `tests/` suite (this delivery's verification was
  sandboxed and shown inline above, not committed as pytest files, since no
  `tests/` directory was uploaded to add them to).
- A full end-to-end train/generate smoke test on a real corpus (would need
  your actual data; the fixes were verified with synthetic data instead).
