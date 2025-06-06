
# solid-lamp
Appen hjælper dig med at få overblik over personer i en roman ved hjælp af kunstig intelligens.

This repository now includes two example scripts.

- `prototype.py` demonstrates a basic chat interface that answers questions about
  the text found up to a detected location in the book.
- `retrieval_qa.py` builds a small Q&A tool that indexes the entire EPUB using
  OpenAI embeddings with FAISS for fast context retrieval.

If `app-main/Kongetro.epub.zip` is present, the prototype can also detect the page number by searching the provided text snippet in the book. Otherwise it falls back to placeholder pages.

## Setup

Create a `.env` file in the project root containing your OpenAI API key:

```
OPENAI_API_KEY=<your_key>
```

You can also copy `.devcontainer/.env.example` to `.env` as a starting point. The
example file contains a placeholder key `OPENAI_API_KEY=sk` that you should
replace with your own.

The `prototype.py` script uses this key to query GPT-4 for answers about the book based on the text up to a detected snippet.

This project expects the `openai` Python package version 1.0 or newer. If you
have an older installation, upgrade with:

```
pip install -U openai
```
