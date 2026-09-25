"""
gamax1/instruction_data.py
===========================
Loads instruction/Q&A-style training data (JSON or JSONL files) for the
fine-tuning stage, as opposed to bulk_corpus.py's plain-.txt continuous-
stream pretraining pipeline.

Key differences from bulk pretraining that this module exists to handle:
  1. Each record is a DISCRETE example (a question + its answer), not a
     slice of one long continuous document -- so examples are padded
     into batches rather than windowed out of a token stream.
  2. Loss must be computed ONLY on the answer/assistant tokens, never on
     the question/prompt tokens -- otherwise the model spends capacity
     learning to predict the *question*, which is not the training goal
     and dilutes the (already scarce) signal we're trying to concentrate
     into it. This is the reason for `loss_mask` throughout this file.

FORMAT DETECTION
-----------------
Your data's actual key names haven't been confirmed yet, so this loader
tries several common conventions, in this order, per record:
  1. {"messages": [{"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}, ...]}
     -- OpenAI/ChatML-style multi-turn. Every assistant turn becomes one
     training example, with all prior turns in that conversation as its
     prompt context (so a 4-turn conversation yields 2 training examples:
     one predicting the first assistant reply, one predicting the second
     with both prior turns as context).
  2. {"instruction": "...", "input": "...", "output": "..."}
     -- Alpaca-style. "input" is optional; if present it's appended to
     "instruction" (matching the standard Alpaca prompt template).
  3. {"question": "...", "answer": "..."}  or
     {"prompt": "...", "response": "..."}  or
     {"prompt": "...", "completion": "..."}
     -- Plain Q&A pairs, whichever key names your files use.

If a record matches NONE of these, it's skipped and counted, and the
final summary tells you how many were skipped along with the key names
seen -- so a real format mismatch is loud and diagnosable, not a silent
zero-example dataset.

If your actual file uses different key names than all of the above,
tell me the exact keys and I'll add a fourth pattern rather than you
having to reshape the data.
"""

import json
import os
import random

import torch


def _iter_records(path: str):
    """Yield dict records from a single .json or .jsonl file.

    .json: either a single object, or a list of objects.
    .jsonl: one JSON object per non-empty line.
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8") as f:
        if ext == ".jsonl":
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] {path}:{line_no}: skipping malformed JSON line ({e})")
        else:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            elif isinstance(data, dict):
                # Some exports wrap the list under a top-level key, e.g.
                # {"data": [...]} or {"examples": [...]}. Try the common
                # ones before giving up and treating the dict as one record.
                for key in ("data", "examples", "records", "conversations"):
                    if key in data and isinstance(data[key], list):
                        for item in data[key]:
                            yield item
                        return
                yield data
            else:
                raise ValueError(f"{path}: top-level JSON must be an object or a list")


def iter_files(data_path: str):
    """Yield every .json/.jsonl file under data_path (a file or a directory,
    recursive)."""
    if os.path.isfile(data_path):
        yield data_path
        return
    for root, _dirs, files in os.walk(data_path):
        for name in sorted(files):
            if name.lower().endswith((".json", ".jsonl")):
                yield os.path.join(root, name)


def _extract_pairs(record: dict):
    """Return a list of (prompt_text, answer_text) pairs from one record,
    per the format-detection rules documented in the module docstring.
    Empty list if the record matches no known format.
    """
    if not isinstance(record, dict):
        return []

    # -- 1. ChatML-style multi-turn messages -------------------------------
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        pairs = []
        context = []
        for msg in messages:
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", ""))
            if role == "user":
                context.append(("user", content))
            elif role == "assistant":
                if context:  # only train on replies that have a preceding prompt
                    prompt_text = "\n".join(c for _, c in context)
                    pairs.append((prompt_text, content))
                context.append(("assistant", content))
            # "system" (or anything else) is folded into context text but
            # not itself a role tag we emit -- GamaX1's tokenizer only
            # reserves <|user|>/<|assistant|>, not a system tag.
            elif role == "system":
                context.append(("system", content))
        return pairs

    # -- 2. Alpaca-style instruction/input/output ---------------------------
    if "instruction" in record and "output" in record:
        instruction = str(record["instruction"])
        extra_input = str(record.get("input", "") or "")
        prompt_text = f"{instruction}\n\n{extra_input}" if extra_input else instruction
        return [(prompt_text, str(record["output"]))]

    # -- 3. Plain Q&A pairs, several common key-name conventions ------------
    key_pairs = (
        ("question", "answer"),
        ("prompt", "response"),
        ("prompt", "completion"),
        ("input", "output"),
    )
    for prompt_key, answer_key in key_pairs:
        if prompt_key in record and answer_key in record:
            return [(str(record[prompt_key]), str(record[answer_key]))]

    return []


def load_pairs(data_path: str):
    """Walk data_path (file or directory) and return (pairs, stats).

    pairs: list of (prompt_text, answer_text) strings, ready to encode.
    stats: dict with counts, useful to sanity-check a format mismatch
    before spending time encoding/training on zero real examples.
    """
    pairs = []
    seen_files = 0
    total_records = 0
    skipped_records = 0
    unmatched_keys_seen = set()

    for path in iter_files(data_path):
        seen_files += 1
        for record in _iter_records(path):
            total_records += 1
            record_pairs = _extract_pairs(record)
            if not record_pairs:
                skipped_records += 1
                if isinstance(record, dict):
                    unmatched_keys_seen.add(tuple(sorted(record.keys())))
                continue
            pairs.extend(record_pairs)

    stats = {
        "files": seen_files,
        "records": total_records,
        "pairs": len(pairs),
        "skipped_records": skipped_records,
        "unmatched_key_sets": list(unmatched_keys_seen)[:5],  # a few examples, not all
    }
    return pairs, stats


def encode_example(tokenizer, prompt_text: str, answer_text: str, max_len: int):
    """Build one training example: token ids plus a same-length loss mask.

    ids  = [<|user|>] + encode(prompt) + [<|assistant|>] + encode(answer) + [<|eos|>]
    mask = [0]*(len up to and including <|assistant|>) + [1]*(answer + eos)

    mask==1 marks exactly the tokens the loss should be computed on --
    the assistant's own answer and the closing eos, never the question
    or the role tags themselves. Truncation, if the example is too long
    for max_len, always removes from the START of the PROMPT first (the
    least important tokens to keep), never from the answer -- an
    example is only answer-truncated (with a printed warning) if the
    answer plus both role tags and eos alone still exceeds max_len.
    """
    user_id = tokenizer.user_id
    assistant_id = tokenizer.assistant_id
    eos_id = tokenizer.eos_id

    prompt_ids = tokenizer.encode(prompt_text)
    answer_ids = tokenizer.encode(answer_text)

    fixed_overhead = 3  # <|user|>, <|assistant|>, <|eos|>
    answer_budget = max(max_len - fixed_overhead, 0)
    if len(answer_ids) > answer_budget:
        print(f"[WARNING] answer alone ({len(answer_ids)} tokens) exceeds --max_len={max_len}; "
              "truncated. Consider raising --max_len for this dataset.")
        answer_ids = answer_ids[:answer_budget]

    prompt_budget = max_len - fixed_overhead - len(answer_ids)
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-max(prompt_budget, 0):]  # keep the END of the prompt (nearest the question)

    ids = [user_id] + prompt_ids + [assistant_id] + answer_ids + [eos_id]
    mask = [0] * (2 + len(prompt_ids)) + [1] * (len(answer_ids) + 1)
    return ids, mask


class InstructionDataset(torch.utils.data.Dataset):
    """Wraps a list of (prompt_text, answer_text) pairs, encoding lazily
    (on __getitem__) so a huge dataset doesn't need every example
    tokenized up front."""

    def __init__(self, pairs, tokenizer, max_len: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt_text, answer_text = self.pairs[idx]
        ids, mask = encode_example(self.tokenizer, prompt_text, answer_text, self.max_len)
        return ids, mask


def make_collate_fn(pad_id: int):
    """Right-pads a batch of variable-length (ids, mask) examples to the
    batch's own max length (not a fixed max_len), which keeps compute
    proportional to what's actually in the batch. Returns xb, yb, loss_mask
    -- all (batch, seq-1) since xb/yb are the standard next-token shift.
    Padded positions get loss_mask==0 automatically (pad_id tokens are
    never real answer content), so they contribute nothing to the loss
    without needing a separate attention-padding mask: GamaX1's attention
    is plain causal self-attention with no padding-mask input, so a
    padded key position CAN be attended to by real tokens before it in
    the same row -- harmless here only because every pad token is placed
    strictly after that row's real content (right-padding) and pad
    positions never contribute to the loss themselves, so their influence
    on earlier positions' predictions is the only leakage. Left-padding
    would leak in a way that matters and must not be used with this
    collate function.
    """

    def collate(batch):
        max_len_in_batch = max(len(ids) for ids, _ in batch)
        batch_x, batch_y, batch_mask = [], [], []
        for ids, mask in batch:
            pad_len = max_len_in_batch - len(ids)
            padded_ids = ids + [pad_id] * pad_len
            padded_mask = mask + [0] * pad_len
            batch_x.append(padded_ids[:-1])
            batch_y.append(padded_ids[1:])
            batch_mask.append(padded_mask[1:])  # mask aligns with the TARGET (yb) position
        return (
            torch.tensor(batch_x, dtype=torch.long),
            torch.tensor(batch_y, dtype=torch.long),
            torch.tensor(batch_mask, dtype=torch.bool),
        )

    return collate


def split_train_val(pairs, val_fraction: float, seed: int = 0):
    """Deterministic shuffle + split, so repeated runs on the same file
    see the same held-out set (useful for comparing fine-tune runs)."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_fraction)) if pairs else 0
    return pairs[n_val:], pairs[:n_val]
