#!/usr/bin/env python3
"""
Prototype — find where an OCR-extracted snippet occurs in Kongetro.epub
and give its Kindle-style location (‘Loc. X of Y’) plus a tiny Q-A loop.

Put this file in  app-main/prototype.py  and execute:
    python app-main/prototype.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, Tuple

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

# ============================================================================
# Configuration & constants
# ============================================================================

PAGE_TEXT_FROM_PICTURE = """
»Det er et drop off. Det er der ingen tvivl om.« Han pegede. »Se. Der, og igen der.
Noget småt. Et usb-stik, måske? Bemærk, hvordan han ser sig over skulderen.
Det er ren refleks.« Storm og Kampman studerede billederne.
»Så hvad er tesen her? Hvis vi lige zoomer ud engang.« Tom så rundt på forsamlingen.
»I siger, at Krebs var på sporet af en eller anden forbindelse mellem Abassi og
Khavari. At Khavari med andre ord er et MOIS-aktiv. Nu er Krebs så død, formentlig
for iranske hænder. Hvad er sammenhængen mellem Krebs og Khavari?«
»Sammenhængen er mord,« sagde Kampman. Tom kiggede op fra billederne. »Siden
Krebs henledte vores opmærksomhed på Khavari, har vi overvåget ham og kæresten,
Julie Severin, i håbet om, at et eller andet brugbart skulle dukke op. Det har
været en ren ørkenvandring. Men så i sidste uge skete der endelig noget. Theo
faldt over en sær forespørgsel i forbindelse med nogle flybilletter, Severin
havde bestilt hos SAS. Dubai-København med afgang fredag aften. En bruger havde
om onsdagen spurgt til afgangen via flyselskabets chatbot. Theo fik sporet
brugeren og kunne konstatere, at den var blevet oprettet på en bogcafé i Dubai
samme dag, og
""".strip()

WORDS_PER_PAGE     = 250   # still useful for chapter estimate
WORDS_PER_LOCATION = 40    # Kindle rule-of-thumb

# Simple mapping (rough)  chapter -> (start_page, end_page)
CHAPTER_RANGES: Dict[int, Tuple[int, int]] = {
    1: (1, 14),
    2: (15, 29),
    3: (30, 44),
    4: (45, 59),
    5: (60, 74),
    6: (75, 89),
    7: (90, 100),
}

# ============================================================================
# Utility helpers
# ============================================================================


def extract_book_text() -> str:
    """Return the book’s plain text (best-effort) or '' if epub missing."""
    zip_path = Path(__file__).with_name("Kongetro.epub.zip")
    if not zip_path.exists():
        return ""

    texts: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith((".xhtml", ".html", ".htm")):
                with zf.open(name) as file:
                    html = file.read().decode("utf-8", errors="ignore")
                    plain = re.sub(r"<[^>]+>", " ", html)  # naïve tag strip
                    texts.append(plain)
    return " ".join(texts)


def _clean(text: str) -> list[str]:
    """Lower-case, remove punctuation, collapse whitespace → list of words."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def locate_snippet(snippet: str, book_text: str) -> int:
    """
    Return the *word index* where `snippet` starts in `book_text`.

    Strategy:
    1) Try exact substring search (fast).
    2) Otherwise do fuzzy search with RapidFuzz.
    """
    # ---------- 1) exact ---------------------------------------------------
    idx = book_text.lower().find(snippet.lower().strip())
    if idx != -1:
        return len(re.findall(r"\w+", book_text[:idx]))

    # ---------- 2) fuzzy ---------------------------------------------------
    snippet_words = _clean(snippet)
    book_words    = _clean(book_text)
    if not snippet_words or not book_words:
        raise ValueError("Empty text given for fuzzy search")

    win_len = min(max(len(snippet_words), 30), 120)  # 30–120-word window
    target  = " ".join(snippet_words[:win_len])

    candidates = [
        " ".join(book_words[i : i + win_len])
        for i in range(len(book_words) - win_len)
    ]

    idx, score, _ = process.extractOne(
        target,
        candidates,
        scorer=Levenshtein.normalized_similarity,
    )

    if score * 100 < 85:
        raise ValueError("Snippet not found (even with fuzzy search)")

    return idx  # absolute word index


def word_index_to_location(word_idx: int) -> int:
    """Approximate Kindle location number from a 0-based word index."""
    return word_idx // WORDS_PER_LOCATION + 1


def answer_question(question: str, context: str) -> str:
    """Very small rule-based Q-A: handles 'Who is NAME?'."""
    m = re.match(r"who is ([^?]+)\?", question.strip(), re.IGNORECASE)
    if not m:
        return "I can only answer questions like 'Who is NAME?' in this prototype."

    name = m.group(1).strip()
    sentences = re.split(r"[.!?]", context)
    hits = [s.strip() for s in sentences if name.lower() in s.lower()]

    return " ".join(hits) + "." if hits else f"I couldn't find information about {name} so far."


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    image_text = PAGE_TEXT_FROM_PICTURE
    book_text  = extract_book_text()

    if not book_text:
        print("Kongetro.epub.zip not found — cannot calculate location.")
        return

    # ------------------------------------------------------------------ locate
    try:
        word_idx = locate_snippet(image_text, book_text)
    except ValueError as exc:
        print(exc)
        return

    # ------------------------------------------------------------------ output
    location        = word_index_to_location(word_idx)
    total_locations = word_index_to_location(len(re.findall(r"\w+", book_text)))

    print(f"Loc. {location} of {total_locations}")  # FIRST output line

    page_estimate = word_idx // WORDS_PER_PAGE + 1
    chapter = next(
        (c for c, (start, end) in CHAPTER_RANGES.items() if start <= page_estimate <= end),
        "unknown"
    )
    print(f"(≈ page {page_estimate}, chapter {chapter})\n")

    # ------------------------------------------------------------------ context
    context_words = _clean(book_text)[: word_idx + WORDS_PER_PAGE]
    context       = " ".join(context_words)

    # ------------------------------------------------------------------ Q-A loop
    print("Ask questions about the story (type 'exit' to quit).")
    while True:
        try:
            q = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if q.lower() in {"exit", "quit"}:
            break
        print("AI:", answer_question(q, context))


if __name__ == "__main__":
    main()
