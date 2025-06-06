#!/usr/bin/env python3
"""Terminal book Q&A using embedding retrieval and OpenAI's API."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Sequence, Dict

import faiss
import numpy as np
import openai
from dotenv import load_dotenv
import ebooklib
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
        if isinstance(item, epub.EpubHtml) or item.get_type() == ebooklib.ITEM_DOCUMENT:
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
    query: str,
    client: openai.OpenAI,
    index: faiss.IndexFlatIP,
    chunks: List[str],
    top_k: int = 3,
) -> List[str]:
    """Return top-k chunk texts most relevant to the query."""
    q_emb = client.embeddings.create(input=query, model=EMBED_MODEL).data[0].embedding
    vec = np.array(q_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    _D, I = index.search(vec, top_k)
    return [chunks[i] for i in I[0]]


def answer_question(
    client: openai.OpenAI,
    question: str,
    context_chunks: List[str],
    model: str = CHAT_MODEL,
    history: Sequence[Dict[str, str]] | None = None,
) -> str:
    """Call ChatCompletion with context and question using optional history."""
    context = "\n".join(ch.strip() for ch in context_chunks)
    system_msg = (
        "You are a helpful assistant answering questions about a book. "
        "Use only the provided context. If the answer is not contained in the context, say you do not know."
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    messages = [{"role": "system", "content": system_msg}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal book Q&A")
    parser.add_argument("--chunk-size", type=int, default=300, help="Token count per chunk")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--model", type=str, default=CHAT_MODEL, help="ChatGPT model")
    args = parser.parse_args()
    # Load .env from ../.devcontainer/.env (relative to this file)
    env_path = Path(__file__).resolve().parents[1] / '.devcontainer' / '.env'
    if env_path.exists():
        from dotenv import dotenv_values
        env = dotenv_values(env_path)
        api_key = env.get("OPENAI_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY missing in environment or in {env_path}")
    client = openai.OpenAI(api_key=api_key)

    epub_path = Path(__file__).with_name("Kongetro.epub.zip")
    try:
        text = extract_text(epub_path)
    except FileNotFoundError:
        print("EPUB file not found; place 'Kongetro.epub.zip' in app-main.")
        return
    if not text.strip():
        print(
            "\u274c No readable text extracted from the EPUB. "
            "Check the file path or extraction logic."
        )
        return


    print("Chunking book text ...")
    chunks = chunk_text(text, max_tokens=args.chunk_size)
    print(f"Total chunks: {len(chunks)}")

    print("Embedding chunks (this may take a while) ...")
    embeds = embed_chunks(client, chunks)
    index = build_index(embeds)

    history: List[Dict[str, str]] = []
    print("Ask a question about the book (type 'exit' to quit).")
    while True:
        try:
            q = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if q.strip().lower() in {"exit", "quit"}:
            break
        top = retrieve(q, client, index, chunks, top_k=args.top_k)
        answer = answer_question(client, q, top, model=args.model, history=history)
        print("AI:", answer)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
