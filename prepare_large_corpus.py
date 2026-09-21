"""
prepare_large_corpus.py (v2 -- multi-book combiner)
======================================================
Downloads a curated list of long, well-known public-domain novels from
Project Gutenberg, strips each one's boilerplate, and concatenates
them into a single large training corpus for GamaX1.

Run this on your own machine (needs internet access):
    python prepare_large_corpus.py                     # default range: Gutenberg IDs 1-400
    python prepare_large_corpus.py --gutenberg_id 100   # single book, unchanged v1 behavior
    python prepare_large_corpus.py --ids 2600 100 1400 2701   # your own list of IDs

A default curated list of long public-domain novels/plays is provided
below, chosen for length and lexical variety (mixed authors/genres
helps a language model generalize better than many similar texts).
"""

import argparse
import re
import time
import urllib.request

START_MARKERS = [
    re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL),
]
END_MARKERS = [
    re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL),
]

# Curated list of long, well-known, public-domain novels (Gutenberg IDs),
# chosen for length + author/genre variety. Rough combined size: several
# million words when all succeed.
DEFAULT_BOOK_IDS = [
    100,    # Complete Works of William Shakespeare
    98,     # A Tale of Two Cities
    84,     # Frankenstein
    76,     # Adventures of Huckleberry Finn
    1342,   # Pride and Prejudice
    1400,   # Great Expectations
    145,    # Middlemarch
    1661,   # The Adventures of Sherlock Holmes
    1952,   # The Yellow Wallpaper
    2554,   # Crime and Punishment
    2600,   # War and Peace
    2701,   # Moby-Dick
    4300,   # Ulysses
    5200,   # Metamorphosis
    1260,   # Jane Eyre
    1497,   # The Republic
    2000,   # Don Quixote
    74,     # The Adventures of Tom Sawyer
    11,     # Alice's Adventures in Wonderland
    16,     # Peter Pan
    35,     # The Time Machine
    36,     # The War of the Worlds
    43,     # The Strange Case of Dr Jekyll and Mr Hyde
    46,     # A Christmas Carol
    55,     # The Wonderful Wizard of Oz
    73,     # The Red Badge of Courage
    120,    # Treasure Island
    1232,   # The Prince
    158,    # Emma
    161,    # Sense and Sensibility
    209,    # The Turn of the Screw
    2148,   # The Odyssey
    219,    # Heart of Darkness
    236,    # The Jungle Book
    244,    # A Study in Scarlet
    27827,  # The Kama Sutra
    28054,  # The Brothers Karamazov
    2814,   # Dubliners
    30254,  # Beyond Good and Evil
    3207,   # Leviathan
    3300,   # The Bible, King James Version
    345,    # Dracula
    34901,  # On Liberty
    3600,   # The Scarlet Letter
    408,    # The Souls of Black Folk
    4217,   # A Portrait of the Artist as a Young Man
    4363,   # Beyond Good and Evil (alt edition)
    514,    # Little Women
    521,    # Paradise Lost
    6130,   # The Iliad
    6133,   # The Aeneid
    730,    # Oliver Twist
    768,    # Wuthering Heights
    829,    # Gulliver's Travels
    863,    # The Mysterious Affair at Styles
    8800,   # Siddhartha
    996,    # Don Juan
    10,     # The King James Bible
    45,     # Anne of Green Gables
    1080,   # A Modest Proposal
    108,    # The Return of Sherlock Holmes
    1399,   # Anna Karenina
    174,    # The Picture of Dorian Gray
    1998,   # Thus Spoke Zarathustra
    205,    # Walden
    2147,   # The Works of Edgar Allan Poe
    26184,  # Simple Sabotage Field Manual
    27805,  # The Art of War
    3206,   # The Federalist Papers
    3825,   # Pygmalion
    5000,   # The Notebooks of Leonardo da Vinci
    55201,  # Meditations
    61,     # The Communist Manifesto
    74,     # Tom Sawyer
    768,    # Wuthering Heights
    902,    # The Happy Prince
    962,    # The Last of the Mohicans
    103,    # Around the World in Eighty Days
    164,    # Twenty Thousand Leagues Under the Seas
    2781,   # The Divine Comedy
    4363,   # Nietzsche Collection
    7370,   # Second Treatise of Government
    7371,   # Essay Concerning Human Understanding
    7372,   # Civil Government
    500,    # The Moonstone
    5740,   # The Art of Money Getting
    5744,   # Essays of Michel de Montaigne
    4368,   # The Golden Bough
    600,    # Notes from Underground
    2448,   # Candide
    1934,   # The Secret Garden
    215,    # The Call of the Wild
    910,    # White Fang
]

# Default to 400 Gutenberg IDs. Unavailable or non-book IDs are skipped by
# the downloader below and reported in the final summary. The curated list
# above remains available as a reference for hand-selecting known works.
DEFAULT_BOOK_IDS = list(range(1, 401))


def strip_gutenberg_boilerplate(text: str) -> str:
    for pattern in START_MARKERS:
        m = pattern.search(text)
        if m:
            text = text[m.end():]
            break
    for pattern in END_MARKERS:
        m = pattern.search(text)
        if m:
            text = text[:m.start()]
            break
    return text.strip()


def download_book(gutenberg_id: int) -> str:
    urls_to_try = [
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
    ]
    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read().decode("utf-8", errors="ignore")
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Download and combine multiple Gutenberg books for GamaX1 training.")
    parser.add_argument("--gutenberg_id", type=int, default=None,
                         help="Download a single book by ID (overrides --ids / default list).")
    parser.add_argument("--ids", type=int, nargs="+", default=None,
                         help="Download and combine a custom list of Gutenberg IDs.")
    parser.add_argument("--out", type=str, default="data/sample_corpus_combined.txt")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Seconds to wait between downloads, polite to Gutenberg's servers (default: 1.0).")
    args = parser.parse_args()

    if args.gutenberg_id is not None:
        book_ids = [args.gutenberg_id]
    elif args.ids is not None:
        book_ids = args.ids
    else:
        book_ids = DEFAULT_BOOK_IDS

    combined_parts = []
    succeeded, failed = [], []

    for i, gid in enumerate(book_ids):
        print(f"[{i+1}/{len(book_ids)}] Downloading Gutenberg ID {gid} ...")
        raw = download_book(gid)
        if raw is None:
            print(f"  FAILED to download ID {gid} (skipping)")
            failed.append(gid)
            continue
        cleaned = strip_gutenberg_boilerplate(raw)
        if len(cleaned) < 1000:
            print(f"  WARNING: cleaned text for ID {gid} looks suspiciously short ({len(cleaned)} chars), skipping")
            failed.append(gid)
            continue
        combined_parts.append(cleaned)
        succeeded.append(gid)
        print(f"  OK: {len(cleaned):,} characters")
        if i < len(book_ids) - 1:
            time.sleep(args.delay)

    if not combined_parts:
        print("\nNo books downloaded successfully. Check your internet connection, or download")
        print("plain-text (.txt) files manually from https://www.gutenberg.org/ and run:")
        print("  python prepare_large_corpus.py --ids <space-separated Gutenberg IDs>")
        return

    combined_text = "\n\n".join(combined_parts)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(combined_text)

    print(f"\n{'='*70}")
    print(f"Combined {len(succeeded)}/{len(book_ids)} books successfully.")
    if failed:
        print(f"Failed IDs (skipped): {failed}")
    print(f"Saved combined corpus to {args.out}")
    print(f"Total size: {len(combined_text):,} characters (~{len(combined_text.split()):,} words)")
    print(f"{'='*70}")
    print(f"\nNow train with, e.g.:")
    print(f"  python -m gamax1.train --tokenizer word --data {args.out} --auto_size_model --max_steps 3000")


if __name__ == "__main__":
    main()
