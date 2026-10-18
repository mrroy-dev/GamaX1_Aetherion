"""
gamax1/knowledge_qa.py
=======================
Verified retrieval-first Q&A over the GamaX1 QnA knowledge base, with a
fine-tuned-model fallback for questions not confidently found in the data.

WHY THIS EXISTS: fine-tuning GamaX1 (a ~98M-param model) directly on ~3,724
unique, arbitrary trivia facts (indian_polity / world_geography / science)
did not produce reliable exact-fact recall, across many attempts (plain
fine-tune, extended epochs, data augmentation, replay-regularized runs --
see finetune-QNA-v1 through v6). This is a data/model-scale mismatch, not a
training bug: the model learned Q&A *format* and topical association well,
but could not bind thousands of specific low-frequency facts into its
weights from a handful of exposures each.

Embedding-based retrieval over the same QnA data solves the FACTUAL-RECALL
problem directly (the fact is looked up, not memorized), but naive
embedding similarity alone is NOT enough: template-similar questions about
different entities (e.g. "capital of India" vs "capital of Mauritius", or
"President of India" vs "the fourth President of India") can score high
cosine similarity despite asking about a completely different fact. Three
extra checks close that gap:
    1. proper-noun entity check   -- query's named entities (country,
       person, place) must all appear in the candidate question.
    2. ordinal/number check       -- numbers/ordinals (e.g. "fourth",
       a year) must match EXACTLY between query and candidate, since an
       extra qualifier like this changes which specific fact is being asked.
    3. topic-word overlap check   -- the remaining content words (the
       actual subject of the question -- "capital" vs "largest state",
       "president" vs "election") must overlap enough; this catches cases
       where the only shared entity is a country name that appears across
       hundreds of unrelated facts in this dataset.
Verified in isolation against known false-positive cases before being
adopted (see conversation history for the test cases this logic passed).

USAGE (fresh Colab session):

    from gamax1.knowledge_qa import load_knowledge_base, smart_answer

    embedder, question_embeddings, records = load_knowledge_base()
    result = smart_answer(
        "Which river forms the Sundarbans Delta?",
        embedder, question_embeddings, records,
    )
    print(result)
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Paths -- adjust here if the project layout changes
# ----------------------------------------------------------------------
KB_DIR = Path("/content/drive/MyDrive/Aetherion_GamaX1/knowledge_base")
FALLBACK_CKPT = Path(
    "/content/drive/MyDrive/Aetherion_GamaX1/runs/gamax1/finetune-QNA-v6/"
    "gamax1_finetune_best.pt"
)
REPO_DIR = Path("/content/gamax1_project")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ----------------------------------------------------------------------
# Lexical verification helpers
# ----------------------------------------------------------------------
SENTENCE_STARTERS = {
    "what", "which", "who", "whom", "whose", "in", "how", "when", "where",
    "why", "is", "are", "do", "does",
}
ORDINAL_WORDS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth",
}
LINK_WORDS = {
    "of", "is", "are", "the", "a", "an", "in", "to", "by", "and", "or",
    "that", "this", "for", "as", "with", "from", "into", "about", "there",
    "was", "were",
}
# Excluded from the topic-word check specifically (not from entity check):
# in this dataset "India"/"Indian" appears in a large fraction of unrelated
# questions, so letting it count toward topic-overlap would make that check
# too easy to pass without actually sharing a real topic.
COUNTRY_SELF = {"india", "indian"}


def extract_proper_nouns(text: str) -> set[str]:
    """Capitalized words, excluding the sentence-initial word (capitalized
    only by position) and common question-starters."""
    words = re.findall(r"[A-Za-z]+", text)
    entities = set()
    for idx, w in enumerate(words):
        if idx == 0:
            continue
        if w[0].isupper() and w.lower() not in SENTENCE_STARTERS:
            entities.add(w.lower())
    return entities


def extract_ordinals_numbers(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z]+|\d+", text.lower())
    return {w for w in words if w.isdigit() or w in ORDINAL_WORDS}


def extract_topic_words(text: str) -> set[str]:
    """Content words carrying the question's actual subject -- everything
    except sentence-starters, link-words, and the country name itself.
    Light stemming (4-char prefix) so river/rivers, form/forms still align."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    terms = set()
    for i, w in enumerate(words):
        if i == 0 or w in SENTENCE_STARTERS or w in LINK_WORDS or w in COUNTRY_SELF:
            continue
        if len(w) < 3:
            continue
        terms.add(w[:4] if len(w) > 4 else w)
    return terms


# ----------------------------------------------------------------------
# Knowledge-base loading
# ----------------------------------------------------------------------
def build_knowledge_base(qa_dir: Path, save: bool = True):
    """One-time build from raw QnA .jsonl files: loads records, computes
    embeddings, and (by default) persists both to KB_DIR for fast reload
    in future sessions. Only needed when the underlying QnA data changes;
    otherwise use load_knowledge_base()."""
    import json
    import pickle
    from sentence_transformers import SentenceTransformer

    records = []
    for f in sorted(Path(qa_dir).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages", [])
            if len(msgs) == 2:
                records.append({
                    "question": msgs[0]["content"],
                    "answer": msgs[1]["content"],
                    "source": f.name,
                })

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    questions = [r["question"] for r in records]
    question_embeddings = embedder.encode(
        questions, convert_to_numpy=True, show_progress_bar=True,
        normalize_embeddings=True,
    )

    if save:
        KB_DIR.mkdir(parents=True, exist_ok=True)
        np.save(KB_DIR / "question_embeddings.npy", question_embeddings)
        with open(KB_DIR / "records.pkl", "wb") as f:
            pickle.dump(records, f)
        print(f"✓ Saved {len(records)} records + embeddings to {KB_DIR}")

    return embedder, question_embeddings, records


def load_knowledge_base():
    """Fast path for a fresh session: loads pre-computed embeddings/records
    from KB_DIR instead of re-encoding all questions from scratch."""
    import pickle
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    question_embeddings = np.load(KB_DIR / "question_embeddings.npy")
    with open(KB_DIR / "records.pkl", "rb") as f:
        records = pickle.load(f)
    print(f"✓ Loaded {len(records)} records, embeddings shape {question_embeddings.shape}")
    return embedder, question_embeddings, records


# ----------------------------------------------------------------------
# Retrieval with verification
# ----------------------------------------------------------------------
def retrieve_verified(
    query: str,
    embedder,
    question_embeddings: np.ndarray,
    records: list[dict],
    top_k: int = 5,
    min_similarity: float = 0.55,
    topic_threshold: float = 0.4,
):
    """Return the best verified match, or None if nothing passes all three
    checks (entity subset, exact ordinal/number match, topic overlap)."""
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = question_embeddings @ q_emb
    top_idx = np.argsort(-sims)[:top_k]

    q_ent = extract_proper_nouns(query)
    q_num = extract_ordinals_numbers(query)
    q_top = extract_topic_words(query)

    for i in top_idx:
        if sims[i] < min_similarity:
            break
        candidate_q = records[i]["question"]
        c_ent = extract_proper_nouns(candidate_q)
        c_num = extract_ordinals_numbers(candidate_q)
        c_top = extract_topic_words(candidate_q)

        entity_ok = q_ent.issubset(c_ent) if q_ent else True
        number_ok = (q_num == c_num)
        overlap = len(q_top & c_top) / len(q_top | c_top) if (q_top | c_top) else 1.0
        topic_ok = overlap >= topic_threshold

        if entity_ok and number_ok and topic_ok:
            return {
                "question": candidate_q,
                "answer": records[i]["answer"],
                "similarity": float(sims[i]),
                "source": records[i]["source"],
            }
    return None


# ----------------------------------------------------------------------
# Hybrid answer: verified retrieval first, fine-tuned model fallback
# ----------------------------------------------------------------------
def smart_answer(
    query: str,
    embedder,
    question_embeddings: np.ndarray,
    records: list[dict],
    fallback_ckpt: Path = FALLBACK_CKPT,
    repo_dir: Path = REPO_DIR,
):
    """Retrieval-first (high precision, verified), model-fallback for
    anything not confidently found in the knowledge base. The fallback is
    NOT reliable for exact facts (see module docstring) -- it's a
    best-effort generation, not a second source of truth."""
    hit = retrieve_verified(query, embedder, question_embeddings, records)
    if hit:
        return {
            "source": "retrieval",
            "answer": hit["answer"],
            "confidence": round(hit["similarity"], 2),
            "matched_question": hit["question"],
            "dataset_file": hit["source"],
        }

    cmd = [
        sys.executable, "-u", "-m", "gamax1.generate",
        "--ckpt", str(fallback_ckpt), "--prompt", query,
        "--max_new_tokens", "40", "--temperature", "0.6", "--top_k", "30",
        "--repetition_penalty", "1.15", "--chat", "--stop_at_eos",
        "--device", "cuda",
    ]
    r = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True)
    raw = r.stdout
    reply = (
        raw.split("<|assistant|>")[-1].replace("<|eos|>", "").strip()
        if "<|assistant|>" in raw else raw.strip()
    )
    return {
        "source": "model (fallback, no confident retrieval match)",
        "answer": reply,
        "confidence": None,
        "matched_question": None,
        "dataset_file": None,
    }
