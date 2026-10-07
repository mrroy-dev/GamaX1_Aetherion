"""
gamax1/instruction_data.py
===========================

Loads instruction/Q&A-style training data (JSON or JSONL files) for the
fine-tuning stage.

This module keeps instruction examples DISCRETE, unlike bulk_corpus.py,
which works with one continuous token stream.

Main responsibilities:

1. Read JSON / JSONL instruction and Q&A datasets.
2. Detect common dataset formats.
3. Convert records into (prompt, answer) pairs.
4. Encode examples with explicit:
       <|user|>
       <|assistant|>
       <|eos|>
   boundaries.
5. Create an answer-only loss mask.
6. Batch variable-length examples with right-padding.
7. Provide deterministic train/validation splitting.

IMPORTANT TRAINING RULE
-----------------------

The loss is calculated ONLY on assistant answer tokens and EOS.

Question/prompt tokens:
    loss_mask = 0

<|user|>:
    loss_mask = 0

<|assistant|>:
    loss_mask = 0

Answer tokens:
    loss_mask = 1

<|eos|>:
    loss_mask = 1

This keeps fine-tuning focused on learning the desired answer behavior
rather than spending the fine-tuning signal on reproducing the question.
"""

import json
import os
import random

import torch


# ======================================================================
# JSON / JSONL loading
# ======================================================================

def _iter_records(path: str):
    """
    Yield dictionary records from a JSON or JSONL file.

    Supported:

        .jsonl
            One JSON object per non-empty line.

        .json
            A single object.

        .json
            A list of objects.

        .json
            A wrapper such as:

                {
                    "data": [...]
                }

            or:

                {
                    "examples": [...]
                }

            or:

                {
                    "records": [...]
                }

            or:

                {
                    "conversations": [...]
                }
    """

    ext = os.path.splitext(path)[1].lower()

    with open(path, encoding="utf-8") as f:

        # --------------------------------------------------------------
        # JSONL
        # --------------------------------------------------------------
        if ext == ".jsonl":

            for line_no, line in enumerate(f, start=1):

                line = line.strip()

                if not line:
                    continue

                try:
                    yield json.loads(line)

                except json.JSONDecodeError as e:

                    print(
                        f"[WARNING] {path}:{line_no}: "
                        f"skipping malformed JSON line ({e})"
                    )

            return

        # --------------------------------------------------------------
        # JSON
        # --------------------------------------------------------------
        data = json.load(f)

        if isinstance(data, list):

            for item in data:
                yield item

            return

        if isinstance(data, dict):

            # Common wrapper formats.
            for key in (
                "data",
                "examples",
                "records",
                "conversations",
            ):

                if key in data and isinstance(data[key], list):

                    for item in data[key]:
                        yield item

                    return

            # Otherwise treat the dictionary itself as one record.
            yield data
            return

        raise ValueError(
            f"{path}: top-level JSON must be an object or a list"
        )


# ======================================================================
# File discovery
# ======================================================================

def iter_files(data_path: str):
    """
    Yield every JSON/JSONL file under data_path.

    data_path may be:

        - a single JSON file
        - a single JSONL file
        - a directory

    Directory traversal is recursive and filenames are processed in
    deterministic sorted order.
    """

    if os.path.isfile(data_path):

        yield data_path
        return

    if not os.path.isdir(data_path):

        raise FileNotFoundError(
            f"Instruction data path does not exist: {data_path}"
        )

    for root, _dirs, files in os.walk(data_path):

        for name in sorted(files):

            if name.lower().endswith(
                (".json", ".jsonl")
            ):
                yield os.path.join(root, name)


# ======================================================================
# Record → prompt / answer pairs
# ======================================================================

def _extract_pairs(record: dict):
    """
    Convert one dataset record into a list of:

        (prompt_text, answer_text)

    Supported formats:

    1. ChatML-style:

        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

    2. Alpaca-style:

        {
            "instruction": "...",
            "input": "...",
            "output": "..."
        }

    3. Plain Q&A:

        {
            "question": "...",
            "answer": "..."
        }

    4. Prompt / response:

        {
            "prompt": "...",
            "response": "..."
        }

    5. Prompt / completion:

        {
            "prompt": "...",
            "completion": "..."
        }

    6. Input / output:

        {
            "input": "...",
            "output": "..."
        }

    Unknown formats return an empty list.
    """

    if not isinstance(record, dict):
        return []

    # ==================================================================
    # 1. ChatML / messages format
    # ==================================================================

    messages = record.get("messages")

    if isinstance(messages, list) and messages:

        pairs = []

        # Context contains all previous turns.
        context = []

        for msg in messages:

            if not isinstance(msg, dict):
                continue

            role = str(
                msg.get("role", "")
            ).lower()

            content = str(
                msg.get("content", "")
            )

            # ----------------------------------------------------------
            # User message
            # ----------------------------------------------------------

            if role == "user":

                context.append(
                    ("user", content)
                )

            # ----------------------------------------------------------
            # Assistant message
            # ----------------------------------------------------------

            elif role == "assistant":

                # Only create a training pair when there is a preceding
                # prompt/context.
                if context:

                    prompt_text = "\n".join(
                        content_text
                        for _, content_text in context
                    )

                    pairs.append(
                        (
                            prompt_text,
                            content,
                        )
                    )

                # Keep assistant answer in context so later assistant
                # turns can see the previous conversation.
                context.append(
                    ("assistant", content)
                )

            # ----------------------------------------------------------
            # System message
            # ----------------------------------------------------------

            elif role == "system":

                context.append(
                    ("system", content)
                )

        return pairs

    # ==================================================================
    # 2. Alpaca format
    # ==================================================================

    if (
        "instruction" in record
        and "output" in record
    ):

        instruction = str(
            record["instruction"]
        )

        extra_input = str(
            record.get("input", "") or ""
        )

        if extra_input:

            prompt_text = (
                f"{instruction}\n\n"
                f"{extra_input}"
            )

        else:

            prompt_text = instruction

        answer_text = str(
            record["output"]
        )

        return [
            (
                prompt_text,
                answer_text,
            )
        ]

    # ==================================================================
    # 3. Plain Q&A / prompt-response formats
    # ==================================================================

    key_pairs = (
        ("question", "answer"),
        ("prompt", "response"),
        ("prompt", "completion"),
        ("input", "output"),
    )

    for prompt_key, answer_key in key_pairs:

        if (
            prompt_key in record
            and answer_key in record
        ):

            return [
                (
                    str(record[prompt_key]),
                    str(record[answer_key]),
                )
            ]

    # Unknown format.
    return []


# ======================================================================
# Dataset loader
# ======================================================================

def load_pairs(data_path: str):
    """
    Walk data_path and return:

        pairs, stats

    pairs:
        list of (prompt_text, answer_text)

    stats:
        diagnostic information useful for checking that the dataset
        was actually understood correctly before fine-tuning.
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

                    unmatched_keys_seen.add(
                        tuple(
                            sorted(
                                record.keys()
                            )
                        )
                    )

                continue

            pairs.extend(
                record_pairs
            )

    stats = {
        "files": seen_files,
        "records": total_records,
        "pairs": len(pairs),
        "skipped_records": skipped_records,
        "unmatched_key_sets": list(
            unmatched_keys_seen
        )[:5],
    }

    return pairs, stats


# ======================================================================
# Encode one instruction example
# ======================================================================

def encode_example(
    tokenizer,
    prompt_text: str,
    answer_text: str,
    max_len: int,
):
    """
    Encode one prompt/answer pair.

    Final structure:

        <|user|>
        prompt tokens
        <|assistant|>
        answer tokens
        <|eos|>

    Token structure:

        ids =
            [
                user_id,
                prompt_tokens...,
                assistant_id,
                answer_tokens...,
                eos_id,
            ]

    Loss mask:

        [
            0,                 # <|user|>
            0, 0, 0, ...       # prompt
            0,                 # <|assistant|>
            1, 1, 1, ...       # answer
            1,                 # <|eos|>
        ]

    IMPORTANT
    ---------

    The tokenizer is responsible for ordinary text tokenization.

    Special role/EOS IDs are inserted MANUALLY here.

    That means fine-tuning examples do not depend on literal special-token
    parsing inside tokenizer.encode().

    Truncation policy:

        1. Preserve the answer whenever possible.
        2. If answer itself is too long, truncate answer and warn.
        3. Otherwise, if the whole example is too long, remove tokens
           from the START of the prompt.
        4. Never remove the answer just to preserve an unnecessarily
           long prompt.
    """

    # --------------------------------------------------------------
    # Stable special-token IDs from the checkpoint tokenizer.
    # --------------------------------------------------------------

    user_id = tokenizer.user_id
    assistant_id = tokenizer.assistant_id
    eos_id = tokenizer.eos_id

    # --------------------------------------------------------------
    # Encode ordinary prompt and answer text.
    # --------------------------------------------------------------

    prompt_ids = tokenizer.encode(
        prompt_text
    )

    answer_ids = tokenizer.encode(
        answer_text
    )

    # --------------------------------------------------------------
    # Three structural tokens:
    #
    #   <|user|>
    #   <|assistant|>
    #   <|eos|>
    # --------------------------------------------------------------

    fixed_overhead = 3

    # Maximum answer tokens that can fit.
    answer_budget = max(
        max_len - fixed_overhead,
        0,
    )

    # --------------------------------------------------------------
    # If answer itself is too long, truncate it.
    # --------------------------------------------------------------

    if len(answer_ids) > answer_budget:

        print(
            "[WARNING] answer alone "
            f"({len(answer_ids)} tokens) exceeds "
            f"--max_len={max_len}; truncated. "
            "Consider raising --max_len for this dataset."
        )

        answer_ids = answer_ids[
            :answer_budget
        ]

    # --------------------------------------------------------------
    # Remaining space belongs to prompt.
    # --------------------------------------------------------------

    prompt_budget = max(
        max_len
        - fixed_overhead
        - len(answer_ids),
        0,
    )

    # --------------------------------------------------------------
    # Keep the END of a long prompt.
    #
    # The end is generally closest to the actual question/current
    # user turn and therefore more useful than the oldest context.
    # --------------------------------------------------------------

    if len(prompt_ids) > prompt_budget:

        if prompt_budget > 0:

            prompt_ids = prompt_ids[
                -prompt_budget:
            ]

        else:

            prompt_ids = []

    # --------------------------------------------------------------
    # Construct final sequence.
    #
    # IMPORTANT:
    # Special IDs are inserted directly rather than encoded as literal
    # strings.
    # --------------------------------------------------------------

    ids = (
        [user_id]
        + prompt_ids
        + [assistant_id]
        + answer_ids
        + [eos_id]
    )

    # --------------------------------------------------------------
    # Answer-only loss mask.
    #
    # <|user|>             -> 0
    # prompt               -> 0
    # <|assistant|>        -> 0
    # answer               -> 1
    # <|eos|>              -> 1
    # --------------------------------------------------------------

    mask = (
        [0] * (
            2 + len(prompt_ids)
        )
        + [1] * (
            len(answer_ids) + 1
        )
    )

    # --------------------------------------------------------------
    # Internal consistency check.
    # --------------------------------------------------------------

    if len(ids) != len(mask):

        raise RuntimeError(
            "Internal instruction encoding error: "
            f"len(ids)={len(ids)} != "
            f"len(mask)={len(mask)}"
        )

    if len(ids) > max_len:

        raise RuntimeError(
            "Internal instruction encoding error: "
            f"encoded length {len(ids)} exceeds "
            f"max_len={max_len}"
        )

    return ids, mask


# ======================================================================
# PyTorch Dataset
# ======================================================================

class InstructionDataset(torch.utils.data.Dataset):
    """
    Wrap a list of (prompt_text, answer_text) pairs.

    Encoding is lazy: examples are tokenized only when __getitem__()
    requests them.

    This avoids keeping an already-tokenized copy of the entire
    fine-tuning dataset in RAM.
    """

    def __init__(
        self,
        pairs,
        tokenizer,
        max_len: int,
    ):

        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx):

        prompt_text, answer_text = (
            self.pairs[idx]
        )

        ids, mask = encode_example(
            self.tokenizer,
            prompt_text,
            answer_text,
            self.max_len,
        )

        return ids, mask


# ======================================================================
# Collate / batch construction
# ======================================================================

def make_collate_fn(pad_id: int):
    """
    Create a right-padding collate function.

    Input:

        [
            (ids, mask),
            (ids, mask),
            ...
        ]

    Output:

        xb
        yb
        loss_mask

    All outputs use the standard causal next-token shift:

        xb = sequence[:-1]
        yb = sequence[1:]

    Therefore the loss mask is shifted in exactly the same way:

        loss_mask = mask[1:]

    Padding:

        Right-padding is used.

        Example:

            [real real real]
            [real real real pad pad]

        Padding receives loss_mask=0.

    IMPORTANT:

    GamaX1 currently uses causal self-attention without an explicit
    padding attention mask.

    Right-padding is therefore intentional here.

    Real answer tokens occur BEFORE the padding positions, so causal
    attention cannot look forward from a real answer token into future
    padding.

    Left-padding must NOT be introduced without changing the model's
    attention masking behavior.
    """

    def collate(batch):

        if not batch:

            raise ValueError(
                "Cannot collate an empty batch"
            )

        # --------------------------------------------------------------
        # Find the longest example in THIS batch.
        # --------------------------------------------------------------

        max_len_in_batch = max(
            len(ids)
            for ids, _ in batch
        )

        batch_x = []
        batch_y = []
        batch_mask = []

        # --------------------------------------------------------------
        # Right-pad every example.
        # --------------------------------------------------------------

        for ids, mask in batch:

            pad_len = (
                max_len_in_batch
                - len(ids)
            )

            padded_ids = (
                ids
                + [pad_id] * pad_len
            )

            padded_mask = (
                mask
                + [0] * pad_len
            )

            # Standard causal shift.
            x = padded_ids[:-1]
            y = padded_ids[1:]

            # Mask must align with TARGET y.
            loss_mask = padded_mask[1:]

            batch_x.append(x)
            batch_y.append(y)
            batch_mask.append(loss_mask)

        # --------------------------------------------------------------
        # Convert to PyTorch tensors.
        # --------------------------------------------------------------

        return (
            torch.tensor(
                batch_x,
                dtype=torch.long,
            ),
            torch.tensor(
                batch_y,
                dtype=torch.long,
            ),
            torch.tensor(
                batch_mask,
                dtype=torch.bool,
            ),
        )

    return collate


# ======================================================================
# Deterministic train / validation split
# ======================================================================

def split_train_val(
    pairs,
    val_fraction: float,
    seed: int = 0,
):
    """
    Deterministically shuffle and split pairs.

    The same:

        dataset
        val_fraction
        seed

    produces the same train/validation split.

    This is important for comparing multiple fine-tuning runs fairly.
    """

    pairs = list(pairs)

    # Deterministic local RNG.
    rng = random.Random(seed)

    rng.shuffle(pairs)

    if not pairs:

        return [], []

    n_val = max(
        1,
        int(
            len(pairs)
            * val_fraction
        ),
    )

    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    return (
        train_pairs,
        val_pairs,
    )