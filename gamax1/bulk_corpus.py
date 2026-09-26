"""Incremental token-cache support for large directories of text books.

The normal trainer is intentionally simple and reads one text file into RAM.
This module is the large-corpus path: it samples text to train a tokenizer,
then encodes each book one at a time into an int32 binary file.  The finished
file is memory mapped, so training batches do not require an 11 GB Python
string or an equally large list of token IDs.

This version supports multiple source categories (e.g. books, wiki, qna) held
in separate sub-directories under one root data directory.  Each source's
files are tracked separately in the metadata so downstream tooling can
compute per-category token counts and, later, per-category evaluation.

Encoding is tracked per file (path + size), not just as one
all-or-nothing corpus manifest. This means adding a brand-new source folder
(or a handful of new files to an existing one) only encodes the new/changed
files and appends them to the token stream -- files that were already
encoded, under the same tokenizer, are left untouched. This matters most on
free-tier Colab, where re-encoding tens of thousands of already-done files
just because one new folder was added would burn most of the session's
compute budget on repeated work. A tokenizer change (different vocab size or
merges) still forces a full rebuild, since every existing token ID would be
wrong under new merges -- nothing can be salvaged in that case.

Encoding itself is also resumable within one incremental batch. Every
``PROGRESS_INTERVAL`` files, the current progress (which files in this
batch are done, and the full per-file token index so far) is written to a
small progress JSON file. If the process is interrupted (e.g. a Colab
disconnect) and re-launched against the same data/cache directories with the
same new/changed file set, encoding picks up right after the last completed
file instead of starting the batch over.
"""

from __future__ import annotations

import json
import mmap
import os
import re
import time
from array import array
from pathlib import Path
from typing import Optional

import torch

from .tokenizer import BPETokenizer


# Explicit dataset paths used by the default corpus build.
DATASETS = {
    "books_cleaned_v1": "/content/drive/MyDrive/Aetherion_GamaX1/data/books_cleaned_v1",
    "Math_Reasoning_train": "/content/drive/MyDrive/Aetherion_GamaX1/data/Math_Reasoning/train/books",
    "Conversations-200k_clean": "/content/drive/MyDrive/Aetherion_GamaX1/data/Conversations-200k_clean",
    "Q&A": "/content/drive/MyDrive/Aetherion_GamaX1/data/QnA",
}

DEFAULT_SOURCE_DIRS = DATASETS

# -- Per-source content format ----------------------------------------------
#
# Different source folders hold genuinely different kinds of text, and
# treating them identically at encoding time is itself a source of the
# "model drifts into an unrelated pattern mid-generation" problem: a plain
# document-boundary token is not enough when a single *file* also glues
# together several unrelated mini-conversations (observed in
# Discord-Dialogues, "---"-separated), or when the file's own speaker
# labels don't map to a real user/assistant role (observed in
# Reddit-Constructive's "Speaker 0:"/"Speaker 1:" format, where role
# identity isn't fixed across threads).
#
# Three formats are supported, chosen deliberately per source rather than
# guessed from content:
#   "prose"         -- continuous text (books/wiki). Gutenberg license
#                      boilerplate is stripped (it is legal filler, not
#                      content, and was otherwise the single most-repeated
#                      pattern across thousands of book files). One
#                      eos_id-bounded unit per file.
#   "user_assistant" -- text with genuine, known "User:"/"Assistant:"
#                      labels. Each turn is wrapped with the tokenizer's
#                      dedicated <|user|>/<|assistant|> ids instead of the
#                      literal text "User:"/"Assistant:" (which would just
#                      be ordinary, spoofable BPE tokens). A file that
#                      glues multiple unrelated exchanges together with a
#                      standalone "---" line is first split on that
#                      separator, and each resulting exchange gets its own
#                      eos_id boundary -- fixing the file-level-only
#                      boundary's blind spot for glued-together turns.
#   "generic_turns" -- dialogue with ambiguous/unfixed speaker identity
#                      (e.g. "Speaker 0:"/"Speaker 1:", where the same
#                      label doesn't reliably mean the same role across
#                      threads). Deliberately does NOT fabricate
#                      <|user|>/<|assistant|> role tags here -- an
#                      incorrect role label would be a worse training
#                      signal than no role label. Still split on a
#                      standalone "---" line, with eos_id between each
#                      resulting turn/exchange, and text encoded plainly.
#
# A source not listed here defaults to "prose" (the safe, unopinionated
# choice) with a one-time warning -- see _resolve_format().
SOURCE_FORMAT_PROSE = "prose"
SOURCE_FORMAT_USER_ASSISTANT = "user_assistant"
SOURCE_FORMAT_GENERIC_TURNS = "generic_turns"
CORPUS_FORMAT_VERSION = "v4-schema-aware-json-size-check"

DEFAULT_SOURCE_FORMATS = {
    "books_cleaned_v1": SOURCE_FORMAT_PROSE,
    "Math_Reasoning_train": SOURCE_FORMAT_USER_ASSISTANT,
    "Conversations-200k_clean": SOURCE_FORMAT_USER_ASSISTANT,
    "Q&A": SOURCE_FORMAT_USER_ASSISTANT,
}

_unknown_source_format_warned: set[str] = set()


def _resolve_format(source_name: str, source_formats: dict) -> str:
    fmt = source_formats.get(source_name)
    if fmt is not None:
        return fmt
    if source_name not in _unknown_source_format_warned:
        print(
            f"[WARNING] No content format configured for source '{source_name}' -- "
            f"defaulting to '{SOURCE_FORMAT_PROSE}' (plain text, one eos_id-bounded "
            "unit per file, no speaker tags). Add it to DEFAULT_SOURCE_FORMATS if "
            "it actually contains dialogue."
        )
        _unknown_source_format_warned.add(source_name)
    return SOURCE_FORMAT_PROSE


# Matches a "---" (or longer) line on its own, with only whitespace around
# it -- the separator observed gluing unrelated Discord/Reddit exchanges
# together. Deliberately requires the WHOLE line to be dashes so it does
# not fire on a literal "---" appearing mid-sentence in real prose.
_STANDALONE_SEP_RE = re.compile(r"(?m)^[ \t]*-{3,}[ \t]*$")

# Recognizes a "User:"/"Assistant:" turn label at the start of a line (or
# start of text) and captures everything up to the next such label. Case
# matches the observed corpus convention exactly; extend the alternation
# here if a source uses different capitalization.
_TURN_RE = re.compile(
    r"(?m)^[ \t]*(User|Assistant)\s*:\s*(.*?)(?=(?:\n[ \t]*(?:User|Assistant)\s*:)|\Z)",
    re.DOTALL,
)

# Standard Project Gutenberg boilerplate markers (same pattern used in
# prepare_large_corpus.py). Keeping content strictly between these two
# markers removes the repeated "The Project Gutenberg eBook of ... This
# eBook is for the use of anyone..." license preamble/footer that would
# otherwise be the single most over-represented pattern across a
# multi-thousand-book corpus.
_GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL
)
_GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL
)


def _strip_gutenberg_boilerplate(text: str) -> str:
    m = _GUTENBERG_START_RE.search(text)
    if m:
        text = text[m.end():]
    m = _GUTENBERG_END_RE.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def _split_on_separator(text: str) -> list[str]:
    """Split on a standalone "---" line into non-empty, stripped chunks."""
    parts = [p.strip() for p in _STANDALONE_SEP_RE.split(text)]
    return [p for p in parts if p]



def _json_records(raw_text: str, suffix: str) -> list[dict]:
    """Parse JSON/JSONL objects conservatively and tolerate common wrappers."""
    records: list[dict] = []
    if suffix.lower() == ".jsonl":
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
            elif isinstance(obj, list):
                records.extend(x for x in obj if isinstance(x, dict))
        return records

    try:
        obj = json.loads(raw_text)
    except json.JSONDecodeError:
        return []
    if isinstance(obj, dict):
        # Common dataset wrappers: {"data": [...]} / {"examples": [...]} etc.
        for key in ("data", "records", "examples", "items", "rows"):
            value = obj.get(key)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return value
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _content_to_text(content) -> str:
    """Normalize string or common multimodal-content representations."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _message_pairs_from_record(record: dict) -> list[tuple[str, str]]:
    """Extract user/assistant pairs from common conversation/Q&A schemas."""
    messages = record.get("messages")
    if isinstance(messages, list):
        pairs: list[tuple[str, str]] = []
        pending_user: str | None = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = _content_to_text(message.get("content", ""))
            if not content:
                continue
            if role in {"user", "human", "question"}:
                pending_user = content
            elif role in {"assistant", "bot", "answer", "gpt", "model"} and pending_user is not None:
                pairs.append((pending_user, content))
                pending_user = None
        if pairs:
            return pairs

    # Common single-turn Q&A schemas, including Johnson-style datasets.
    user_keys = ("question", "prompt", "query", "instruction", "input", "user", "human")
    assistant_keys = ("answer", "response", "output", "completion", "assistant", "bot", "target")
    user_text = next((_content_to_text(record.get(k)) for k in user_keys if _content_to_text(record.get(k))), "")
    assistant_text = next((_content_to_text(record.get(k)) for k in assistant_keys if _content_to_text(record.get(k))), "")
    if user_text and assistant_text:
        return [(user_text, assistant_text)]

    # Some datasets store an explicit two-turn list under conversation/dialog.
    for key in ("conversation", "dialog", "dialogue", "turns"):
        turns = record.get(key)
        if not isinstance(turns, list):
            continue
        pending_user = None
        pairs = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", turn.get("speaker", ""))).strip().lower()
            content = _content_to_text(turn.get("content", turn.get("text", turn.get("value", ""))))
            if not content:
                continue
            if role in {"user", "human", "question"}:
                pending_user = content
            elif role in {"assistant", "bot", "answer", "model", "gpt"} and pending_user is not None:
                pairs.append((pending_user, content))
                pending_user = None
        if pairs:
            return pairs
    return []


def _extract_json_training_text(raw_text: str, suffix: str) -> list[str]:
    """Extract actual textual training content for BPE sampling."""
    chunks: list[str] = []
    for record in _json_records(raw_text, suffix):
        pairs = _message_pairs_from_record(record)
        if pairs:
            for user_text, assistant_text in pairs:
                chunks.extend((user_text, assistant_text))
            continue
        # For prose-like JSON records, use explicit text/content fields only.
        for key in ("text", "content", "document", "body"):
            value = _content_to_text(record.get(key))
            if value:
                chunks.append(value)
                break
    return chunks


def _encode_json_messages(tokenizer: BPETokenizer, raw_text: str, suffix: str) -> list[int]:
    """Encode JSON/JSONL Q&A records as reserved user/assistant role tokens."""
    ids: list[int] = []
    for record in _json_records(raw_text, suffix):
        for user_text, assistant_text in _message_pairs_from_record(record):
            if ids:
                ids.append(tokenizer.eos_id)
            ids.append(tokenizer.user_id)
            ids.extend(tokenizer.encode(user_text))
            ids.append(tokenizer.assistant_id)
            ids.extend(tokenizer.encode(assistant_text))
    return ids


def _encode_user_assistant_block(tokenizer: BPETokenizer, block: str) -> list[int]:
    """Encode one exchange, replacing literal "User:"/"Assistant:" labels
    with the tokenizer's dedicated role ids. Falls back to plain prose
    encoding if no recognizable turn label is found at all, since forcing
    a role tag onto untagged text would be a fabricated signal."""
    matches = list(_TURN_RE.finditer(block))
    if not matches:
        return tokenizer.encode(block)

    ids = []
    preamble = block[:matches[0].start()].strip()
    if preamble:
        ids.extend(tokenizer.encode(preamble))
    for m in matches:
        role, content = m.group(1), m.group(2).strip()
        ids.append(tokenizer.user_id if role == "User" else tokenizer.assistant_id)
        if content:
            ids.extend(tokenizer.encode(content))
    return ids


def _encode_source_file(
    tokenizer: BPETokenizer, source_name: str, raw_text: str, source_formats: dict,
    is_first_emission: bool, suffix: str = "",
) -> tuple[list[int], bool]:
    """Encode one file's text according to its source's content format.

    Returns ``(token_ids, is_first_emission)`` where the returned
    ``is_first_emission`` has been updated for the caller's next file --
    threading it through this way keeps the "does this need a leading
    eos_id" decision correct across BOTH file boundaries and any
    within-file "---"-separated boundaries this function introduces,
    without the caller needing to know how many sub-blocks a file split
    into.
    """
    fmt = _resolve_format(source_name, source_formats)
    ids: list[int] = []

    def emit(block_ids: list[int]):
        nonlocal is_first_emission
        if not is_first_emission:
            ids.append(tokenizer.eos_id)
        is_first_emission = False
        ids.extend(block_ids)

    if fmt == SOURCE_FORMAT_PROSE:
        emit(tokenizer.encode(_strip_gutenberg_boilerplate(raw_text)))
    elif fmt == SOURCE_FORMAT_USER_ASSISTANT:
        if suffix.lower() in {".json", ".jsonl"}:
            json_ids = _encode_json_messages(tokenizer, raw_text, suffix)
            if json_ids:
                emit(json_ids)
        else:
            for block in (_split_on_separator(raw_text) or [raw_text]):
                emit(_encode_user_assistant_block(tokenizer, block))
    elif fmt == SOURCE_FORMAT_GENERIC_TURNS:
        for block in (_split_on_separator(raw_text) or [raw_text]):
            emit(tokenizer.encode(block))
    else:  # pragma: no cover -- _resolve_format never returns anything else
        emit(tokenizer.encode(raw_text))

    return ids, is_first_emission


# Write a progress checkpoint every this many files during encoding.
# Corpus encoding checkpoint frequency.
#
# Why 500? Encoding thousands of files can take a long time on a Drive-mounted
# Colab filesystem. Every 500 completed files we flush the token stream and
# write `encode_progress.json`. If Colab disconnects after that point, the next
# run can resume instead of starting the whole corpus again.
#
# This is a FILE checkpoint, not a neural-network training checkpoint.
ENCODE_CHECKPOINT_EVERY = 500

# Backwards-compatible name used by older notebook/debugging code. Keeping the
# alias avoids breaking a cell that still prints or inspects PROGRESS_INTERVAL.
PROGRESS_INTERVAL = ENCODE_CHECKPOINT_EVERY

# Every this many files, force an OS-level fsync (not just a Python-level
# flush) and pause briefly while checking that the on-disk file size has
# stopped changing -- a practical proxy for "Google Drive's background sync
# has likely caught up". Meant for split sessions (e.g. a daily 4-hour GPU
# quota against a 5-hour encode): stopping the runtime right after one of
# these hard-sync points is much safer than stopping between them.
HARD_SYNC_INTERVAL = 10_000
HARD_SYNC_STABLE_CHECKS = 3   # consecutive stable size readings required
HARD_SYNC_CHECK_DELAY_SEC = 10  # seconds between size checks


def _format_duration(seconds: float) -> str:
    """Return compact human-readable elapsed time for progress logs."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _hard_sync_checkpoint(output, token_path: Path) -> None:
    """Force an OS-level fsync and wait for the on-disk file size to settle.

    ``output.flush()`` only pushes Python's buffer into the OS -- it does not
    guarantee Google Drive's FUSE mount has pushed those bytes to the actual
    cloud copy. ``os.fsync`` forces the OS to write its buffers to the mount,
    and then re-checking the file size a few times a few seconds apart gives
    a practical (not perfect) signal that Drive's background sync has caught
    up: if the size is still climbing, Drive is still working through a
    backlog and it is not a good time to disconnect.
    """
    output.flush()
    os.fsync(output.fileno())

    stable_count = 0
    last_size = -1
    for _ in range(HARD_SYNC_STABLE_CHECKS + 5):  # bounded, never hangs forever
        current_size = token_path.stat().st_size
        if current_size == last_size:
            stable_count += 1
            if stable_count >= HARD_SYNC_STABLE_CHECKS:
                break
        else:
            stable_count = 0
        last_size = current_size
        time.sleep(HARD_SYNC_CHECK_DELAY_SEC)

    print(
        f"  [hard sync] fsync'd and size stable at {last_size:,} bytes -- "
        f"safe to stop the runtime now if you need to."
    )


class BulkTokenStore:
    """Read-only memory-mapped token storage with a torch tensor view."""

    def __init__(self, token_path: Path, token_count: int):
        self.token_path = Path(token_path)
        self._file = self.token_path.open("rb")
        # ACCESS_COPY gives torch a writable view without copying the entire
        # token file into RAM; writes remain private and never touch the cache.
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)
        self.tensor = torch.frombuffer(self._mapping, dtype=torch.int32, count=token_count)

    def close(self):
        # Release the tensor view before closing its backing mmap.
        self.tensor = None
        self._mapping.close()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _collect_source_paths(data_dir: str | Path, source_dirs=DEFAULT_SOURCE_DIRS) -> dict[str, list[Path]]:
    """Collect text files from explicit dataset paths or legacy subfolders."""
    if isinstance(source_dirs, dict):
        candidates = [(str(name), Path(path)) for name, path in source_dirs.items()]
    else:
        root = Path(data_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"data directory does not exist: {root}")
        candidates = [(str(name), root / name) for name in source_dirs]

    sources: dict[str, list[Path]] = {}
    for name, source_path in candidates:
        if not source_path.is_dir():
            continue
        allowed_suffixes = {".txt", ".json", ".jsonl"}
        paths = sorted(
            (
                p for p in source_path.rglob("*")
                if p.is_file() and p.suffix.lower() in allowed_suffixes
            ),
            key=lambda p: str(p),
        )
        if paths:
            sources[name] = paths

    if not sources:
        root = Path(data_dir)
        if root.is_dir():
            allowed_suffixes = {".txt", ".json", ".jsonl"}
            flat_paths = sorted(
                (
                    p for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in allowed_suffixes
                ),
                key=lambda p: str(p),
            )
            if flat_paths:
                sources["books"] = flat_paths

    if not sources:
        raise ValueError(
            f"no supported source files (.txt/.json/.jsonl) found: {candidates}"
        )
    return sources


def book_paths(data_dir: str | Path) -> list[Path]:
    """Return deterministic, recursive .txt input paths across all sources.

    Kept for backward compatibility with callers that just want a flat list
    of every file regardless of source category. Order is: source category
    name order, then path order within each source.
    """
    sources = _collect_source_paths(data_dir)
    paths: list[Path] = []
    for name in sorted(sources):
        paths.extend(sources[name])
    return paths


def sample_book_text(paths: list[Path], sample_chars: int) -> str:
    """Read actual textual content, not raw JSON syntax, for BPE training."""
    if sample_chars <= 0:
        raise ValueError("sample_chars must be positive")
    pieces: list[str] = []
    remaining = sample_chars
    for path in paths:
        if remaining <= 0:
            break
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            raw = handle.read(remaining * 2 if path.suffix.lower() in {".json", ".jsonl"} else remaining)
        if path.suffix.lower() in {".json", ".jsonl"}:
            extracted = _extract_json_training_text(raw, path.suffix)
            text = "\n\n".join(extracted)
        else:
            text = raw
        if text:
            text = text[:remaining]
            pieces.append(text)
            remaining -= len(text)
    sample = "\n\n".join(pieces)
    if not sample.strip():
        raise ValueError("input files contain no readable training text")
    return sample


def _file_stat(path: Path) -> dict:
    """Cheap identity check for a source file: size only.

    mtime is deliberately NOT used here. Google Drive's FUSE mount does not
    reliably preserve file mtimes across a remount (e.g. after
    drive.flush_and_unmount() at the end of a session) -- a file's reported
    mtime can drift between sessions even though its content never changed.
    That drift changes how many files look "new/changed" from one session
    to the next, and _load_resumable_progress refuses to resume at all when
    that count (``batch_size``) doesn't match the checkpoint -- silently
    forcing a full restart of the batch even when the token cache on disk
    is perfectly fine. Size-only detection sidesteps this: these are static
    text files that don't change in place, so size alone is a reliable
    enough signal, and it isn't affected by Drive's mtime drift.
    """
    stat = path.stat()
    return {"size": stat.st_size}


def _file_index_path(cache_dir: Path) -> Path:
    return cache_dir / "file_index.json"


def _progress_path(cache_dir: Path) -> Path:
    return cache_dir / "encode_progress.json"

def _encoding_timing_path(cache_dir: Path) -> Path:
    return cache_dir / "encoding_checkpoint_timing.jsonl"


def _load_file_index(cache_dir: Path, tokenizer_expected: dict) -> Optional[dict]:
    """Load the per-file token index, or None if absent/unusable.

    Only usable if the tokenizer identity (vocab size + BPE merges) matches
    exactly -- reusing token IDs encoded under different merges would
    silently corrupt the stream, so any tokenizer change forces a clean
    rebuild rather than a partial reuse.
    """
    path = _file_index_path(cache_dir)
    if not path.exists():
        return None
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if index.get("tokenizer") != tokenizer_expected:
        return None
    return index


def _save_file_index(cache_dir: Path, index: dict) -> None:
    _atomic_write_text(
        _file_index_path(cache_dir),
        json.dumps(index, indent=2),
    )


def _load_resumable_progress(
    cache_dir: Path, token_path: Path, tokenizer_expected: dict, batch_size: int,
) -> Optional[dict]:
    """Return a valid in-progress checkpoint for the current incremental
    batch of new/changed files, or None if none applies.

    A checkpoint is only usable if it was written for the exact same
    tokenizer AND the exact same number of files in this batch -- if the set
    of new/changed files differs from what the checkpoint expected (e.g.
    yet another folder was added mid-run), we refuse to resume and let the
    caller restart this batch cleanly instead of risking a misaligned token
    stream.
    """
    progress_file = _progress_path(cache_dir)
    if not progress_file.exists() or not token_path.exists():
        return None
    try:
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if progress.get("tokenizer") != tokenizer_expected:
        return None
    # New files may have appeared after an interrupted run.  If the checkpoint
    # contains its exact batch_paths, those paths define the resumable batch;
    # do not reject the checkpoint merely because the current corpus has more
    # files now.  Older checkpoints without batch_paths retain the old
    # batch-size guard.
    if "batch_paths" not in progress and progress.get("batch_size") != batch_size:
        return None

    expected_bytes = int(progress["token_count"]) * array("I").itemsize
    actual_bytes = token_path.stat().st_size
    if actual_bytes < expected_bytes:
        return None
    if actual_bytes > expected_bytes:
        # A partial extra file may have been written after the last
        # checkpoint before the interruption. Truncate back to the last
        # confirmed-good checkpoint boundary so the token stream stays
        # file-aligned, then resume from there.
        with token_path.open("r+b") as handle:
            handle.truncate(expected_bytes)

    return progress


def _tokenizer_path(cache_dir: Path) -> Path:
    return cache_dir / "tokenizer.json"


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomically replace a small metadata/tokenizer JSON file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def _load_or_create_persistent_bpe(
    cache_dir: Path,
    all_paths: list[Path],
    bpe_vocab_size: int,
    bpe_sample_chars: int,
    supplied: Optional[BPETokenizer],
) -> BPETokenizer:
    """Load the cache-owned tokenizer, creating it exactly once if necessary.

    Training settings such as batch size, LR, epochs/max_steps, dropout, etc.
    never participate in tokenizer creation.  A persistent tokenizer is what
    makes the token IDs stable across Colab sessions and across later runs.
    """
    path = _tokenizer_path(cache_dir)

    if supplied is not None:
        # A training checkpoint may carry the tokenizer.  Persist it so future
        # runs can use the same tokenizer even without a model checkpoint.
        if path.exists():
            try:
                loaded = BPETokenizer.load(path)
                if loaded.merges != supplied.merges:
                    raise ValueError(
                        "Supplied checkpoint tokenizer differs from the tokenizer "
                        "stored in the bulk cache. Use the cache tokenizer or explicitly "
                        "rebuild the corpus cache with a new cache directory."
                    )
                return loaded
            except ValueError:
                raise
            except (OSError, KeyError, json.JSONDecodeError):
                pass
        payload = {"merges": [list(pair) for pair in supplied.merges]}
        _atomic_write_text(path, json.dumps(payload, indent=2))
        return supplied

    if path.exists():
        try:
            tok = BPETokenizer.load(path)
            print(f"Using persistent BPE tokenizer: {path}")
            return tok
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"[WARNING] Could not load {path}: {exc}. Rebuilding tokenizer.")

    print("Training BPE tokenizer once for this bulk cache...")
    tok = BPETokenizer(
        sample_book_text(all_paths, bpe_sample_chars),
        vocab_size=bpe_vocab_size,
        sample_chars=bpe_sample_chars,
    )
    payload = {"merges": [list(pair) for pair in tok.merges]}
    _atomic_write_text(path, json.dumps(payload, indent=2))
    print(f"Saved persistent BPE tokenizer: {path}")
    return tok


def _quarantine_file(path: Path, reason: str = "invalid") -> None:
    """Move a suspect cache file aside instead of deleting it."""
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"{path.name}.{reason}.{stamp}.bak")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.name}.{reason}.{stamp}.{counter}.bak")
        counter += 1
    path.replace(target)
    print(f"[SAFE RECOVERY] Moved {path.name} to {target.name}")


def build_or_load_bulk_tokens(
    data_dir: str | Path,
    cache_dir: str | Path,
    *,
    bpe_vocab_size: int = 16000,
    bpe_sample_chars: int = 3_000_000,
    tokenizer: Optional[BPETokenizer] = None,
    rebuild: bool = False,
    source_dirs=DEFAULT_SOURCE_DIRS,
    source_formats: dict = None,
) -> tuple[BPETokenizer, BulkTokenStore, dict]:
    """Build/reuse a persistent memory-mapped bulk token cache.

    ``source_formats`` maps each source-folder name to one of
    SOURCE_FORMAT_PROSE / SOURCE_FORMAT_USER_ASSISTANT /
    SOURCE_FORMAT_GENERIC_TURNS (see the module-level comment above
    DEFAULT_SOURCE_FORMATS for what each does and why). Defaults to
    DEFAULT_SOURCE_FORMATS; pass a copy with overrides to customize
    without editing this file.

    IMPORTANT:
      * Changing training settings does NOT invalidate this cache.
      * Existing files are never re-encoded when their size is unchanged.
      * New files are appended only.
      * A changed/deleted previously-encoded file triggers a clean rebuild,
        because this append-only token stream cannot safely replace bytes in
        the middle.
      * The tokenizer is persisted inside ``cache_dir/tokenizer.json`` and is
        therefore stable across Colab sessions.
      * ``encode_progress.json`` resumes an interrupted batch from its last
        checkpoint, even after the runtime disappears.
    """
    if tokenizer is not None and not isinstance(tokenizer, BPETokenizer):
        raise ValueError("bulk training requires a BPETokenizer")

    if source_formats is None:
        source_formats = DEFAULT_SOURCE_FORMATS

    sources = _collect_source_paths(data_dir, source_dirs)
    all_paths = [p for name in sorted(sources) for p in sources[name]]
    path_to_source = {
        str(p): name for name, paths in sources.items() for p in paths
    }

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    token_path = cache / "tokens.int32.bin"
    meta_path = cache / "metadata.json"
    progress_path = _progress_path(cache)

    # The tokenizer belongs to the corpus cache, not to the training run.
    tokenizer = _load_or_create_persistent_bpe(
        cache, all_paths, bpe_vocab_size, bpe_sample_chars, tokenizer
    )

    tokenizer_expected = {
        "tokenizer": "bpe",
        "vocab_size": tokenizer.vocab_size,
        "merges": [list(pair) for pair in tokenizer.merges],
        "corpus_format_version": CORPUS_FORMAT_VERSION,
        "source_formats": dict(sorted(source_formats.items())),
    }

    if rebuild:
        # Explicit rebuild is non-destructive: quarantine old artifacts
        # instead of deleting them.
        for path in (token_path, _file_index_path(cache), progress_path):
            _quarantine_file(path, "explicit_rebuild")
        file_index = None
    else:
        file_index = _load_file_index(cache, tokenizer_expected)

    # An invalid/missing index means the existing binary cannot be trusted.
    # NEVER append to it: that would mix token IDs from different corpus/parser
    # identities. Quarantine the binary and start a clean stream.
    if file_index is None:
        if token_path.exists():
            _quarantine_file(token_path, "cache_identity_changed")
        if progress_path.exists():
            _quarantine_file(progress_path, "cache_identity_changed")
        files_record: dict[str, dict] = {}
        base_token_count = 0
    else:
        files_record = file_index["files"]
        base_token_count = int(file_index["token_count"])

        # Append-only storage cannot remove/replace a file in the middle.
        # If an existing file changed or disappeared, rebuild cleanly.
        current_keys = {str(p) for p in all_paths}
        changed = [
            key for key, record in files_record.items()
            if key not in current_keys or
            _file_stat(Path(key)).get("size") != record.get("size")
        ]
        if changed:
            print(
                f"[WARNING] {len(changed)} previously-encoded file(s) were "
                "changed or removed. Rebuilding the bulk token stream so stale "
                "tokens cannot remain mixed with new content."
            )
            _quarantine_file(token_path, "changed_files")
            _quarantine_file(progress_path, "changed_files")
            files_record = {}
            base_token_count = 0

    # Determine only genuinely new files.
    to_encode = []
    for path in all_paths:
        key = str(path)
        stat = _file_stat(path)
        recorded = files_record.get(key)
        if recorded is None or recorded.get("size") != stat.get("size"):
            to_encode.append(path)

    token_count = base_token_count

    if to_encode:
        # Resume uses the exact file list from the interrupted batch.  This is
        # deliberately independent of the current total corpus file count:
        # if new files appeared after a disconnect, finish the old batch first.
        resume_progress = None if rebuild else _load_resumable_progress(
            cache, token_path, tokenizer_expected, len(to_encode)
        )

        if resume_progress is not None:
            saved_paths = resume_progress.get("batch_paths")
            current_paths = [str(p) for p in to_encode]
            if saved_paths is not None and saved_paths != current_paths:
                # New files were added/reordered. Try to resume the exact old
                # batch instead of throwing away already-encoded progress.
                saved_set = set(saved_paths)
                if not all(p in {str(x) for x in all_paths} for p in saved_paths):
                    resume_progress = None
                else:
                    # The old batch must be contiguous from its saved order;
                    # after it completes, a subsequent invocation will append
                    # newly discovered files.
                    to_encode = [Path(p) for p in saved_paths]
                    resume_progress = _load_resumable_progress(
                        cache, token_path, tokenizer_expected, len(to_encode)
                    )

        if resume_progress is not None:
            start_index = int(resume_progress["files_done"])
            token_count = int(resume_progress["token_count"])
            files_record = resume_progress["files_record"]
            file_mode = "r+b"
            print(
                f"Resuming incremental encode from file "
                f"{start_index:,}/{len(to_encode):,} | "
                f"{token_count:,} tokens in cache so far"
            )
        else:
            start_index = 0
            token_count = base_token_count
            expected_bytes = base_token_count * array("I").itemsize
            if token_path.exists():
                actual_bytes = token_path.stat().st_size
                if actual_bytes != expected_bytes:
                    # Never append to an unverified byte boundary.
                    with token_path.open("r+b") as handle:
                        handle.truncate(expected_bytes)
            file_mode = "ab" if token_path.exists() else "wb"

        batch_paths = [str(p) for p in to_encode]
        with token_path.open(file_mode) as output:
            if file_mode == "r+b":
                output.seek(0, 2)

            encode_run_start_time = time.monotonic()
            last_progress_time = encode_run_start_time
            last_progress_index = start_index
            last_progress_token_count = token_count
            timing_path = _encoding_timing_path(cache)
            # Timing is append-only and survives disconnects. A resumed run
            # starts a new timing interval from its current checkpoint; it never
            # fabricates the time spent before the runtime disappeared.
            timing_records = []
            if timing_path.exists():
                for line in timing_path.read_text(encoding="utf-8").splitlines()[-20:]:
                    try: timing_records.append(json.loads(line))
                    except Exception: pass
            previous_checkpoint_time = None
            previous_checkpoint_index = start_index
            # Tracks whether the very next emitted block (file, or
            # "---"-separated sub-block within a file) needs a leading
            # eos_id. False only for the very first block of the entire
            # cache; True forever after -- including across a resume,
            # since token_count/start_index > 0 there means something
            # was already emitted in an earlier run or earlier file.
            is_first_emission = (token_count == 0 and start_index == 0)

            for index, path in enumerate(
                to_encode[start_index:], start=start_index + 1
            ):
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    raw_text = handle.read()

                source_name = path_to_source[str(path)]
                encoded, is_first_emission = _encode_source_file(
                    tokenizer, source_name, raw_text, source_formats,
                    is_first_emission, path.suffix
                )

                values = array("I", encoded)
                values.tofile(output)

                key = str(path)
                stat = _file_stat(path)
                files_record[key] = {
                    "source": path_to_source[key],
                    "size": stat["size"],
                    "token_start": token_count,
                    "token_count": len(encoded),
                }
                token_count += len(encoded)

                if index % ENCODE_CHECKPOINT_EVERY == 0 or index == len(to_encode):
                    output.flush()
                    os.fsync(output.fileno())

                    # Timing must be computed BEFORE the progress-JSON write below,
                    # since that write embeds interval_elapsed/files_per_sec/eta_sec.
                    # (A prior version of this block referenced these variables in
                    # the JSON dump before they were assigned, which raised
                    # UnboundLocalError on the very first checkpoint of any run.)
                    now = time.monotonic()
                    interval_files = index - last_progress_index
                    interval_tokens = token_count - last_progress_token_count
                    interval_elapsed = now - last_progress_time
                    run_elapsed = now - encode_run_start_time
                    files_per_sec = (
                        interval_files / interval_elapsed
                        if interval_elapsed > 0 else 0.0
                    )
                    tokens_per_sec = (
                        interval_tokens / interval_elapsed
                        if interval_elapsed > 0 else 0.0
                    )
                    remaining_files = max(0, len(to_encode) - index)
                    eta_sec = (remaining_files / files_per_sec) if files_per_sec > 0 else None

                    _atomic_write_text(
                        progress_path,
                        json.dumps(
                            {
                                "tokenizer": tokenizer_expected,
                                "batch_size": len(to_encode),
                                "batch_paths": batch_paths,
                                "files_done": index,
                                "token_count": token_count,
                                "files_record": files_record,
                                "last_checkpoint_wall_time": time.time(),
                                "last_checkpoint_interval_sec": interval_elapsed,
                                "last_checkpoint_files_per_sec": files_per_sec,
                                "eta_seconds_to_batch_end": eta_sec,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                    timing_record = {
                        "timestamp": time.time(), "checkpoint_files": index,
                        "interval_files": interval_files, "interval_sec": interval_elapsed,
                        "files_per_sec": files_per_sec, "tokens_per_sec": tokens_per_sec,
                        "eta_seconds_to_batch_end": eta_sec,
                        "estimated_batch_end_unix": (time.time()+eta_sec if eta_sec is not None else None),
                    }
                    with timing_path.open("a", encoding="utf-8") as tf:
                        tf.write(json.dumps(timing_record, sort_keys=True) + "\n")
                    print(
                        f"Encoded {index:,}/{len(to_encode):,} new files "
                        f"| {token_count:,} tokens total "
                        f"| last {interval_files:,} files: "
                        f"{_format_duration(interval_elapsed)} "
                        f"({files_per_sec:.2f} files/s, "
                        f"{tokens_per_sec:,.0f} tok/s) "
                        f"| run elapsed: {_format_duration(run_elapsed)} "
                        f"| checkpoint saved"
                    )
                    last_progress_time = now
                    last_progress_index = index
                    last_progress_token_count = token_count

                if index % HARD_SYNC_INTERVAL == 0 or index == len(to_encode):
                    print(
                        f"  [hard sync] {index:,} files done -- "
                        "forcing fsync and checking Drive sync..."
                    )
                    _hard_sync_checkpoint(output, token_path)

        if progress_path.exists():
            progress_path.unlink()
    else:
        print(
            f"Reusing bulk token cache: {token_path} "
            f"({token_count:,} tokens) -- no new or changed files"
        )

    source_token_counts: dict[str, int] = {}
    for record in files_record.values():
        source_token_counts[record["source"]] = (
            source_token_counts.get(record["source"], 0) + record["token_count"]
        )

    file_index = {
        "tokenizer": tokenizer_expected,
        "files": files_record,
        "token_count": token_count,
    }
    _save_file_index(cache, file_index)

    metadata = {
        **tokenizer_expected,
        "token_count": token_count,
        "source_token_counts": source_token_counts,
        "file_count": len(files_record),
    }
    _atomic_write_text(
        meta_path,
        json.dumps(metadata, indent=2),
    )

    print(
        f"Bulk corpus ready: {len(files_record):,} files, "
        f"{token_count:,} tokens | by source: {source_token_counts}"
    )

    final_expected_bytes = token_count * array("I").itemsize
    final_actual_bytes = token_path.stat().st_size
    if final_actual_bytes != final_expected_bytes:
        raise RuntimeError(
            f"Bulk token cache is corrupt: {token_path} is "
            f"{final_actual_bytes:,} bytes, but the file index claims "
            f"{token_count:,} tokens ({final_expected_bytes:,} bytes). "
            "Automatic deletion is disabled; inspect the cache and make a backup before any explicit rebuild."
        )

    return tokenizer, BulkTokenStore(token_path, token_count), metadata