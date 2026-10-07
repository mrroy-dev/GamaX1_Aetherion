"""
gamax1/tokenizer.py
====================
Dependency-free tokenizers for GamaX1.

- ``CharTokenizer``: character-level baseline.
- ``WordTokenizer``: word/punctuation level with ``<unk>`` and an
  optional frequency cap.
- ``BPETokenizer``: a GPT-2-style byte-level BPE (subword) tokenizer.
  This is the recommended choice for real corpora: it needs no
  external dependency, never emits ``<unk>`` (any byte sequence is
  representable), and produces far more training tokens per byte than
  word tokenization, which is what actually drives the quality of
  next-token language modeling on a corpus like the bundled one.

All tokenizers expose the same interface (``encode``/``decode``/
``vocab_size``/``save``/``load``) so swapping between them is a one-line
change, and ``generate`` restores the right one from the checkpoint.
"""

import heapq
import json
import re
from collections import Counter


def word_tokenizer_warning(tokenizer_type: str, token_count: int, vocab_size: int):
    """Return a useful warning when a word vocabulary lacks training signal.

    A tiny word corpus gives most words too few repeated contexts to learn a
    next-word distribution. Character tokenization remains a better default in
    that situation, but this is deliberately advisory rather than a blocker.
    """
    if tokenizer_type == "word" and (token_count < 2000 or vocab_size < 200):
        return (
            f"[WARNING] Word-level tokenizer built from only {token_count} "
            f"word occurrences and a vocabulary of {vocab_size} words. "
            "This is likely too small for word-level generation to produce "
            "coherent text -- consider using --tokenizer char instead, "
            "or a larger corpus (--corpus large / a bigger --data file)."
        )
    return None


class CharTokenizer:
    """Simple character-level tokenizer.

    Every character in the training vocabulary maps to one token id.
    Unlike BPE, this tokenizer does not use byte merges or special-token
    machinery. It is intentionally simple and deterministic for baseline
    and smoke-test training.
    """

    def __init__(self, text: str = None, vocab: list = None):
        if vocab is not None:
            self.chars = list(vocab)
        else:
            if text is None:
                raise ValueError("text is required when vocab is not supplied")
            self.chars = sorted(set(text))

        if not self.chars:
            raise ValueError("character vocabulary cannot be empty")

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text: str):
        """Encode each character into its vocabulary id."""
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            ch = exc.args[0]
            raise ValueError(
                f"Character {ch!r} is not present in the tokenizer vocabulary"
            ) from None

    def decode(self, ids):
        """Decode token ids back into characters."""
        try:
            return "".join(self.itos[int(i)] for i in ids)
        except KeyError as exc:
            raise ValueError(
                f"Token id {exc.args[0]} is not present in the tokenizer vocabulary"
            ) from None

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.chars, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            chars = json.load(f)
        return cls(vocab=chars)


class WordTokenizer:
    """Small dependency-free word/punctuation tokenizer.

    Keeping punctuation as its own token lets generated text retain readable
    sentence boundaries while ``<unk>`` makes inference safe for prompts that
    contain vocabulary not seen during training.
    """

    unk_token = "<unk>"
    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(
        self,
        text: str = None,
        vocab: list = None,
        max_vocab_size: int = None,
    ):
        """Build a vocabulary, optionally capped with ``<unk>`` included.

        ``max_vocab_size`` counts every entry, including the reserved ``<unk>``
        token. The remaining slots retain the most frequent source tokens;
        frequency ties are resolved lexically for reproducible checkpoints.
        """
        if vocab is not None:
            self.tokens = vocab
        else:
            if text is None:
                raise ValueError("text is required when vocab is not supplied")
            if max_vocab_size is not None and max_vocab_size < 1:
                raise ValueError(
                    "max_vocab_size must be at least 1 when provided"
                )

            counts = Counter(
                token
                for token in self._tokenize(text)
                if token != self.unk_token
            )

            ranked_tokens = sorted(
                counts,
                key=lambda token: (-counts[token], token),
            )

            if max_vocab_size is not None:
                ranked_tokens = ranked_tokens[: max_vocab_size - 1]

            self.tokens = [self.unk_token] + ranked_tokens

        if self.unk_token not in self.tokens:
            self.tokens.insert(0, self.unk_token)

        self.stoi = {token: i for i, token in enumerate(self.tokens)}
        self.itos = {i: token for i, token in enumerate(self.tokens)}

    @classmethod
    def _tokenize(cls, text: str):
        return cls._pattern.findall(text)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, text: str):
        unk_id = self.stoi[self.unk_token]
        return [
            self.stoi.get(token, unk_id)
            for token in self._tokenize(text)
        ]

    def decode(self, ids):
        tokens = [
            self.itos.get(int(i), self.unk_token)
            for i in ids
        ]

        text = ""
        no_space_before = set(".,!?;:%)]}")
        no_space_after = set("([{")

        for token in tokens:
            if (
                not text
                or token in no_space_before
                or text[-1] in no_space_after
            ):
                text += token
            else:
                text += " " + token

        return text

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.tokens, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            tokens = json.load(f)
        return cls(vocab=tokens)


class BPETokenizer:
    """Dependency-free byte-level BPE tokenizer, GPT-2 style.

    Text is pre-tokenized into words and punctuation (whitespace kept
    attached to the following word), each unit is split into UTF-8
    bytes, and the most frequent adjacent byte pairs are merged until
    ``vocab_size`` is reached.

    Merges never cross word boundaries, which keeps encoding deterministic
    and reproducible.

    The base vocabulary is the 256 byte values; every merge adds one
    token that subsumes a byte sequence. Because any byte sequence is
    representable, ``decode`` never falls back to ``<unk>``.

    ``sample_chars`` caps how much of the training text is scanned for
    pair statistics. Larger values produce a slightly better vocabulary
    but cost linear time in Python.
    """

    _pattern = re.compile(
        r"\s*\d|\s*[^\W\d]+|\s+|[^\w\s]+",
        re.UNICODE,
    )

    min_pair_count = 2
    progress_interval_chars = 50_000_000

    # Reserved IDs.
    #
    # IMPORTANT:
    # Keep this ordering stable because checkpoints depend on these IDs.
    SPECIAL_TOKENS = (
        "<|eos|>",
        "<|user|>",
        "<|assistant|>",
        "<|pad|>",
        "<|think|>",
        "<|/think|>",
    )

    def __init__(
        self,
        text: str = None,
        vocab_size: int = 8000,
        merges: list = None,
        sample_chars: int = 3_000_000,
    ):
        if merges is not None:
            self.merges = [tuple(m) for m in merges]
        else:
            if text is None:
                raise ValueError(
                    "text is required when merges are not supplied"
                )

            if vocab_size < 256:
                raise ValueError(
                    "vocab_size must be at least 256 for byte-level BPE"
                )

            self.merges = self._train(
                text,
                vocab_size,
                sample_chars,
            )

        self._build_vocab()

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def _train(
        self,
        text: str,
        vocab_size: int,
        sample_chars: int,
    ) -> list:
        if sample_chars is not None and len(text) > sample_chars:
            text = text[:sample_chars]

        units = [
            list(unit.encode("utf-8"))
            for unit in self._pattern.findall(text)
        ]

        if not units:
            return []

        counts = Counter()
        occurrences = {}

        for idx, ids in enumerate(units):
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] += 1
                occurrences.setdefault((a, b), set()).add(idx)

        heap = []

        for pair, count in counts.items():
            heapq.heappush(heap, (-count, pair))

        merges = []
        next_id = 256

        while len(merges) < vocab_size - 256:
            pair = None

            while heap:
                neg_count, candidate = heapq.heappop(heap)

                current = counts.get(candidate, 0)

                if current != -neg_count:
                    continue

                if current < self.min_pair_count:
                    pair = None
                    break

                pair = candidate
                break

            if pair is None:
                break

            a, b = pair

            counts.pop(pair, None)

            new_id = next_id
            next_id += 1

            merges.append(pair)

            for idx in list(occurrences.get(pair, ())):
                old_ids = units[idx]

                new_ids = []
                i = 0

                while i < len(old_ids):
                    if (
                        i + 1 < len(old_ids)
                        and old_ids[i] == a
                        and old_ids[i + 1] == b
                    ):
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(old_ids[i])
                        i += 1

                units[idx] = new_ids

                old_counts = Counter(
                    zip(old_ids, old_ids[1:])
                )

                new_counts = Counter(
                    zip(new_ids, new_ids[1:])
                )

                # Pairs newly created by the merge.
                for p, delta in (
                    new_counts - old_counts
                ).items():
                    counts[p] = counts.get(p, 0) + delta

                    occurrences.setdefault(p, set()).add(idx)

                    heapq.heappush(
                        heap,
                        (-counts[p], p),
                    )

                # Pairs removed by the merge.
                for p, delta in (
                    old_counts - new_counts
                ).items():
                    counts[p] -= delta

                    if counts[p] <= 0:
                        counts.pop(p, None)
                    else:
                        heapq.heappush(
                            heap,
                            (-counts[p], p),
                        )

                    occ = occurrences.get(p)

                    if occ is not None:
                        occ.discard(idx)

                        if not occ:
                            del occurrences[p]

            occurrences.pop(pair, None)

        return merges

    def _build_vocab(self):
        # Base byte vocabulary.
        self.token_bytes = {
            i: bytes([i])
            for i in range(256)
        }

        # Every BPE merge receives an ID beginning at 256.
        for new_id, (a, b) in enumerate(
            self.merges,
            start=256,
        ):
            self.token_bytes[new_id] = (
                self.token_bytes[a]
                + self.token_bytes[b]
            )

        self.byte_to_id = {
            b: i
            for i, b in self.token_bytes.items()
        }

        # Trie used for greedy longest-match encoding.
        #
        # Byte values are 0..255.
        # -1 is therefore safe as a terminal marker.
        self._merge_trie = {}

        for token_id, token_bytes in self.token_bytes.items():
            node = self._merge_trie

            for byte in token_bytes:
                node = node.setdefault(byte, {})

            node[-1] = token_id

    # ------------------------------------------------------------------
    # IDs / special tokens
    # ------------------------------------------------------------------

    @property
    def vocab_size(self):
        # 256 base bytes
        # + learned BPE merges
        # + reserved special tokens
        return (
            256
            + len(self.merges)
            + len(self.SPECIAL_TOKENS)
        )

    @property
    def special_token_ids(self):
        """Return the stable IDs of all reserved special tokens."""
        base = 256 + len(self.merges)

        return {
            name: base + i
            for i, name in enumerate(self.SPECIAL_TOKENS)
        }

    @property
    def eos_id(self):
        """Reserved ID marking a document/conversation boundary."""
        return self.special_token_ids["<|eos|>"]

    @property
    def user_id(self):
        """Reserved ID marking the start of a user turn."""
        return self.special_token_ids["<|user|>"]

    @property
    def assistant_id(self):
        """Reserved ID marking the start of an assistant turn."""
        return self.special_token_ids["<|assistant|>"]

    @property
    def pad_id(self):
        """Reserved ID used for padding instruction examples."""
        return self.special_token_ids["<|pad|>"]

    @property
    def think_id(self):
        """Reserved ID marking the beginning of a reasoning trace."""
        return self.special_token_ids["<|think|>"]

    @property
    def think_end_id(self):
        """Reserved ID marking the end of a reasoning trace."""
        return self.special_token_ids["<|/think|>"]

    @property
    def tokens(self):
        ids_to_name = {
            v: k
            for k, v in self.special_token_ids.items()
        }

        out = []

        for i in range(self.vocab_size):
            if i in ids_to_name:
                out.append(ids_to_name[i])
            else:
                out.append(
                    self.token_bytes[i].decode("latin-1")
                )

        return out

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str):
        """
        Encode text while preserving the EXACT order of normal text
        and special tokens.

        Example:

            <|user|>What is Article 17?<|assistant|>Abolition.<|eos|>

        must become:

            [user_id,
             ...question tokens...,
             assistant_id,
             ...answer tokens...,
             eos_id]

        The old implementation collected special-token IDs first and
        ordinary text afterward. That could transform:

            A <|user|> B <|assistant|> C

        into:

            <|user|> <|assistant|> A B C

        which is incorrect.

        This implementation processes every segment in its original
        position, so the token sequence exactly follows the input order.
        """

        special_re = re.compile(
            r"(<\|eos\|>|<\|user\|>|<\|assistant\|>|"
            r"<\|pad\|>|<\|think\|>|<\|/think\|>)"
        )

        special_map = {
            "<|eos|>": self.eos_id,
            "<|user|>": self.user_id,
            "<|assistant|>": self.assistant_id,
            "<|pad|>": self.pad_id,
            "<|think|>": self.think_id,
            "<|/think|>": self.think_end_id,
        }

        ids = []

        # Keep special tokens in the resulting list.
        #
        # Example:
        #
        #   <|user|>Hello<|assistant|>Hi<|eos|>
        #
        # becomes:
        #
        #   ["", "<|user|>", "Hello",
        #    "<|assistant|>", "Hi", "<|eos|>", ""]
        #
        # Each segment is processed immediately, preserving order.
        segments = special_re.split(text)

        for segment in segments:
            if not segment:
                continue

            # Special token:
            # append it RIGHT HERE.
            if segment in special_map:
                ids.append(special_map[segment])
                continue

            # Ordinary text:
            # encode it immediately before moving to the next segment.
            for unit in self._pattern.findall(segment):
                b = unit.encode("utf-8")

                i = 0
                n = len(b)

                while i < n:
                    node = self._merge_trie
                    j = i

                    # Start with the individual byte token.
                    last_id = self.byte_to_id[b[i:i + 1]]
                    last_j = i + 1

                    # Greedy longest-match trie walk.
                    while j < n:
                        nxt = node.get(b[j])

                        if nxt is None:
                            break

                        node = nxt
                        j += 1

                        term = node.get(-1)

                        if term is not None:
                            last_id = term
                            last_j = j

                    ids.append(last_id)
                    i = last_j

        return ids

    def decode(self, ids):
        """
        Decode normal BPE tokens back to UTF-8 text.

        Reserved special tokens have no byte representation and are
        therefore omitted from normal decode().
        """

        special_ids = set(
            self.special_token_ids.values()
        )

        parts = []

        for i in ids:
            i = int(i)

            if i in special_ids:
                # Special tokens are structural, not text bytes.
                parts.append(b"")
            else:
                parts.append(self.token_bytes[i])

        return b"".join(parts).decode(
            "utf-8",
            errors="replace",
        )

    def decode_with_boundaries(
        self,
        ids,
        boundary_marker: str = None,
    ):
        """
        Decode while making special tokens visible.

        Useful for checking whether a training/generation sequence
        contains the expected user/assistant/EOS boundaries.
        """

        ids_to_name = {
            v: k
            for k, v in self.special_token_ids.items()
        }

        parts = []

        for i in ids:
            i = int(i)

            if i in ids_to_name:
                if boundary_marker is not None:
                    tag = boundary_marker
                else:
                    tag = (
                        f"\n\n"
                        f"{ids_to_name[i]}"
                        f"\n\n"
                    )

                parts.append(tag)

            else:
                parts.append(
                    self.token_bytes[i].decode("latin-1")
                )

        # token_bytes are represented as latin-1 so every byte maps
        # one-to-one to one Python character. Convert the complete
        # sequence back to UTF-8 afterward.
        raw = "".join(parts).encode(
            "latin-1",
            errors="ignore",
        )

        return raw.decode(
            "utf-8",
            errors="replace",
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """
        Save only the learned BPE merges.

        Special-token IDs are derived from SPECIAL_TOKENS and the
        number of merges, so they do not need to be stored separately.
        """
        with open(path, "w") as f:
            json.dump(
                {
                    "merges": [
                        list(m)
                        for m in self.merges
                    ]
                },
                f,
            )

    @classmethod
    def load(cls, path: str):
        """Load a previously saved BPE tokenizer."""
        with open(path) as f:
            payload = json.load(f)

        return cls(
            merges=payload["merges"]
        )