#!/usr/bin/env python3
"""Terminal book Q&A using embedding retrieval and OpenAI's API."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

import faiss
import numpy as np
import openai
from dotenv import load_dotenv
from ebooklib import epub
import tiktoken


# ---------------------------------------------------------------------------
# EPUB extraction and chunking
# ---------------------------------------------------------------------------


def extract_text(epub_file: Path) -> str:
    """Return raw text from the given EPUB file."""
    if not epub_file.exists():
        raise FileNotFoundError(f"EPUB not found: {epub_file}")

    book = epub.read_epub(str(epub_file))
    texts: List[str] = []
    for item in book.get_items():
        if item.get_type() == epub.EpubHtml:
            html = item.get_content().decode("utf-8", errors="ignore")
            # naive HTML tag strip
            plain = re.sub(r"<[^>]+>", " ", html)
            texts.append(plain)
    return "\n".join(texts)


def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    """Split text into roughly max_tokens token chunks."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks: List[str] = []
    cur: List[int] = []
    for token in tokens:
        cur.append(token)
        if len(cur) >= max_tokens:
            chunks.append(enc.decode(cur))
            cur = []
    if cur:
        chunks.append(enc.decode(cur))
    return chunks


# ---------------------------------------------------------------------------
# Embedding and index building
# ---------------------------------------------------------------------------

EMBED_MODEL = "text-embedding-ada-002"


def embed_chunks(client: openai.OpenAI, chunks: List[str]) -> np.ndarray:
    """Embed all chunks and return a 2D numpy array."""
    embeddings = []
    batch_size = 16
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=EMBED_MODEL)
        for d in resp.data:
            embeddings.append(d.embedding)
    return np.array(embeddings, dtype="float32")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Return a FAISS index for the given embeddings."""
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# Retrieval and answer generation
# ---------------------------------------------------------------------------

CHAT_MODEL = "gpt-3.5-turbo"


def retrieve(
    query: str, client: openai.OpenAI, index: faiss.IndexFlatIP, chunks: List[str], top_k: int = 3
) -> List[str]:
    """Return top-k chunk texts most relevant to the query."""
    q_emb = client.embeddings.create(input=query, model=EMBED_MODEL).data[0].embedding
    vec = np.array(q_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    _D, I = index.search(vec, top_k)
    return [chunks[i] for i in I[0]]


def answer_question(client: openai.OpenAI, question: str, context_chunks: List[str]) -> str:
    """Call ChatCompletion with context and question."""
    context = "\n".join(ch.strip() for ch in context_chunks)
    system_msg = (
        "You are a helpful assistant answering questions about a book. "
        "Use only the provided context. If the answer is not contained in the context, say you do not know."
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment")
    client = openai.OpenAI(api_key=api_key)

    epub_path = Path(__file__).with_name("Kongetro.epub.zip")
    try:
        text = extract_text(epub_path)
    except FileNotFoundError:
        print("EPUB file not found; place 'Kongetro.epub.zip' in app-main.")
        return

    print("Chunking book text ...")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    print("Embedding chunks (this may take a while) ...")
    embeds = embed_chunks(client, chunks)
    index = build_index(embeds)

    print("Ask a question about the book (type 'exit' to quit).")
    while True:
        try:
            q = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if q.strip().lower() in {"exit", "quit"}:
            break
        top = retrieve(q, client, index, chunks)
        answer = answer_question(client, q, top)
        print("AI:", answer)


if __name__ == "__main__":
    main()
