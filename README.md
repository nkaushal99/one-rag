# One RAG — Step 1

This is the first learning milestone: plain-text documents are chunked, embedded,
stored in Qdrant, and retrieved by semantic similarity. It intentionally has no
framework, API, UI, hybrid search, reranker, or LLM yet.

## Run it

1. Start Qdrant: `docker compose up -d`
2. Install dependencies: `uv sync`
3. Copy `.env.template` to `.env` if it is not already present.
4. Index the sample documents: `uv run rag.py index`
5. Search: `uv run rag.py ask "How can I change my credentials?"`

Add your own UTF-8 `.txt` files under `documents/`, then run the index command again.

## What to inspect

- `chunk_text` shows fixed-size chunking with overlap.
- `SentenceTransformer` turns text into vectors.
- Qdrant ranks the question vector against chunk vectors using cosine similarity.
- The CLI prints retrieved chunks, source, chunk number, and similarity score.

The next step will add an LLM to turn the retrieved chunks into a cited answer.

Design decisions are recorded in [`adr/`](adr/).
