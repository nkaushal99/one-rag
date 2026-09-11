# One RAG — MVP

The project begins with a deliberately small local infrastructure foundation and
a transparent retrieval CLI. It does not use LangChain or LangGraph.

## Stage 0: local infrastructure

`docker compose up --build` starts these local services:

- FastAPI at `http://localhost:8000` (interactive docs: `/docs`)
- PostgreSQL at `localhost:5432` for future document/job metadata
- Qdrant at `localhost:6333` for vectors
- Redis at `localhost:6379` for future cache/short-lived job state
- Redpanda at `localhost:19092`, a Kafka-compatible broker for future ingestion jobs

Verify the stack with `curl http://localhost:8000/health/ready`. It checks all
four dependencies and returns HTTP 503 until they are reachable. Data volumes are
named and persist across container restarts. `docker compose down` stops services;
add `-v` only when you intentionally want to delete local data.

The `.env.template` file uses host-facing addresses for local commands. Compose
overrides those addresses inside the API container. The existing CLI reads the
host-facing `QDRANT_URL` from `.env` (normally `http://localhost:6333`). LLM and reranker settings are
declared but deliberately unset: provider integration belongs to a later stage.

## Stage 1: transparent retrieval

Plain-text documents are chunked, embedded, stored in Qdrant, and retrieved by
semantic similarity. The CLI intentionally has no LLM yet.

## Run it

1. Start the local stack: `docker compose up --build -d`
2. Install CLI retrieval dependencies: `uv sync --extra retrieval`
3. Copy `.env.template` to `.env` if it is not already present.
4. Index the sample documents: `uv run rag.py index`
5. Search: `uv run rag.py ask "How can I change my credentials?"`

Add your own UTF-8 `.txt` files under `documents/`, then run the index command again.

Set `CHUNKING_STRATEGY` in `.env` to `sentence`, `sentence_window`, `section`,
or `paragraph`, then index again before comparing search results. The default
`sentence_window` uses two sentences with one overlapping sentence.

## What to inspect

- `chunk_text` selects one explicit chunking strategy from `.env`.
- `SentenceTransformer` turns text into vectors.
- Qdrant ranks the question vector against chunk vectors using cosine similarity.
- The CLI prints retrieved chunks, source, chunk number, and similarity score.

The next step will add an LLM to turn the retrieved chunks into a cited answer.

Design decisions are recorded in [`adr/`](adr/).
