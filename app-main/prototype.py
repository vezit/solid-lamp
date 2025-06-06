import re
from typing import Dict, Tuple

# Simple mapping of chapter ranges: chapter -> (start_page, end_page)
CHAPTER_RANGES: Dict[int, Tuple[int, int]] = {
    1: (1, 14),
    2: (15, 29),
    3: (30, 44),
    4: (45, 59),
    5: (60, 74),
    6: (75, 89),
    7: (90, 100)
}

# Sample book pages with a few key pages mentioning Theo
SPECIFIC_PAGES = {
    5: "Theo is introduced early as a minor character.",
    81: "At last, Theo appears. Theo is a cunning merchant from the east.",
    82: "Theo helps the hero solve a puzzle.",
    83: "They travel together. Theo proves to be brave and loyal.",
}

def get_page_text(page: int) -> str:
    """Return placeholder text for a given page."""
    if page in SPECIFIC_PAGES:
        return SPECIFIC_PAGES[page]
    return f"Generic page {page} text."


def gather_context(up_to_page: int) -> str:
    """Gather text from page 1 up to `up_to_page`."""
    pages = [get_page_text(p) for p in range(1, up_to_page + 1)]
    return " " .join(pages)


def parse_page_number(image_str: str) -> int:
    """Extract a page number from the given image string."""
    match = re.search(r"page\s*(\d+)", image_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    raise ValueError("Could not detect page number in image string")


def get_chapter(page: int) -> int:
    for chap, (start, end) in CHAPTER_RANGES.items():
        if start <= page <= end:
            return chap
    raise ValueError("Page out of range")


def answer_question(question: str, context: str) -> str:
    """Very small rule-based QA for 'who is NAME?'"""
    m = re.match(r"who is ([^?]+)\?", question.strip(), re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        # Search for sentences mentioning the name
        sentences = re.split(r"[.!?]", context)
        hits = [s.strip() for s in sentences if name.lower() in s.lower()]
        if hits:
            return " ".join(hits) + "."
        else:
            return f"I couldn't find information about {name} so far."
    return "I can only answer questions like 'Who is NAME?' in this prototype."


def main():
    image = input("Paste image text containing the page number: ")
    try:
        page = parse_page_number(image)
    except ValueError as e:
        print(e)
        return
    chapter = get_chapter(page)
    print(f"Detected page {page}, which is in chapter {chapter}.")
    context = gather_context(page)

    print("Ask questions about the story. Type 'exit' to quit.")
    while True:
        q = input("You: ")
        if q.lower() in {"exit", "quit"}:
            break
        answer = answer_question(q, context)
        print("AI:", answer)

if __name__ == "__main__":
    main()
