#!/usr/bin/env python3
"""Simple prototype for tracking characters in a novel using OpenAI."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Set

import psycopg2

import openai
from dotenv import dotenv_values, load_dotenv
import ebooklib
from ebooklib import epub
import tiktoken


# ---------------------------------------------------------------------------
# EPUB extraction and chunking helpers
# ---------------------------------------------------------------------------


def extract_text(epub_file: Path) -> str:
    """Return raw text from the EPUB."""
    if not epub_file.exists():
        raise FileNotFoundError(f"EPUB not found: {epub_file}")

    book = epub.read_epub(str(epub_file))
    texts: List[str] = []
    for item in book.get_items():
        if isinstance(item, epub.EpubHtml) or item.get_type() == ebooklib.ITEM_DOCUMENT:
            html = item.get_content().decode("utf-8", errors="ignore")
            plain = re.sub(r"<[^>]+>", " ", html)
            texts.append(plain)
    return "\n".join(texts)


def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    """Yield chunks of roughly ``max_tokens`` tokens."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    cur: List[int] = []
    chunks: List[str] = []
    for token in tokens:
        cur.append(token)
        if len(cur) >= max_tokens:
            chunks.append(enc.decode(cur))
            cur = []
    if cur:
        chunks.append(enc.decode(cur))
    return chunks


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

CHAT_MODEL = "gpt-3.5-turbo"


def load_api_key() -> str:
    """Load OPENAI_API_KEY from environment or .devcontainer/.env."""
    load_dotenv()
    env_path = Path(__file__).resolve().parents[1] / ".devcontainer" / ".env"
    if env_path.exists():
        env = dotenv_values(env_path)
        key = env.get("OPENAI_API_KEY")
    else:
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            f"OPENAI_API_KEY not found in environment or {env_path}"
        )
    return key


def extract_names(client: openai.OpenAI, text: str) -> List[str]:
    """Ask the model for character names mentioned in the text."""
    system_msg = (
        "You extract names of fictional characters from novel excerpts. "
        "Return a comma-separated list of unique names."
    )
    user_msg = (
        "Text:\n" + text + "\n\n" + "List the character names mentioned:" 
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0,
        max_tokens=60,
    )
    names = resp.choices[0].message.content
    return [n.strip() for n in names.split(",") if n.strip()]


def connect_db():
    """Return a connection to PostgreSQL or ``None`` if unavailable."""
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd = os.getenv("POSTGRES_PASSWORD", "postgres")
    db = os.getenv("POSTGRES_DB", "solidlamp")
    try:
        conn = psycopg2.connect(host=host, port=port, user=user, password=pwd, dbname=db)
        conn.autocommit = True
    except Exception as exc:
        print(f"Database connection failed: {exc}")
        return None
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS characters (id SERIAL PRIMARY KEY, name TEXT UNIQUE)"
        )
    return conn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    key = load_api_key()
    client = openai.OpenAI(api_key=key)

    epub_path = Path(__file__).with_name("Kongetro.epub.zip")
    try:
        text = extract_text(epub_path)
    except FileNotFoundError:
        print(
            "EPUB not found. Place 'Kongetro.epub.zip' alongside this script."
        )
        return

    chunks = chunk_text(text)
    seen: Set[str] = set()

    for chunk in chunks:
        for name in extract_names(client, chunk):
            seen.add(name)

    conn = connect_db()

    print("Characters found:")
    for name in sorted(seen):
        print("-", name)
        if conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO characters(name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                    (name,)
                )
    if conn:
        conn.close()


if __name__ == "__main__":
    main()
