
# solid-lamp
Appen hjælper dig med at få overblik over personer i en roman ved hjælp af kunstig intelligens.

This repository now includes `prototype.py` which demonstrates a simple chat interface for exploring characters in a book.
The Q\&A loop can answer basic questions about the text found up to a detected location.

If `app-main/Kongetro.epub.zip` is present, the prototype can also detect the page number by searching the provided text snippet in the book. Otherwise it falls back to placeholder pages.

## Setup

Create a `.env` file in the project root containing your OpenAI API key:

```
OPENAI_API_KEY=<your_key>
```

The `prototype.py` script uses this key to query GPT-4 for answers about the book based on the text up to a detected snippet.
