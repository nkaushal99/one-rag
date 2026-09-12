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

Plain-text documents are sentence-chunked, embedded with an open-source model,
stored in Qdrant, and retrieved by cosine similarity. The API returns the
retrieved evidence and the exact constructed context; an LLM answer is a later
stage, so no provider key is required to use this MVP.

`POST /v1/documents` accepts `document_id`, `source`, and `text`. Re-submitting
the same `document_id` replaces its existing chunks. `POST /v1/query` accepts a
`question` and optional `limit` (1–10), then returns ranked evidence, similarity
scores, and a context string. Both are available in the running API's `/docs`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/documents -H "Content-Type: application/json" -d '{"document_id":"account-guide","source":"account-guide.txt","text":"Credentials can be modified from Account Settings."}'
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question":"How can I reset my password?"}'
```

An import-ready Postman collection is available at
[`postman/one-rag.postman_collection.json`](postman/one-rag.postman_collection.json).
Import it, then drag the `Health` and `Stage 1 RAG` folders into your `on-rag`
collection if you want to keep that collection as the parent.

If this project contains vectors written by the earlier CLI, migrate every stored
text payload to the current schema with `uv run --extra retrieval
scripts/reindex_qdrant.py`. This rebuilds the Qdrant collection after reading its
stored text, preserving the chunks while adding `document_id`, `source`, and
`chunk_index` to every point.

## Run it

1. Start the local stack: `docker compose up --build -d`
2. Install retrieval dependencies: `uv sync --extra retrieval`
3. Copy `.env.template` to `.env` if it is not already present.
4. Index the sample documents: `uv run --extra retrieval rag.py index`
5. Search: `uv run --extra retrieval rag.py ask "How can I change my credentials?"`

Add your own UTF-8 `.txt` files under `documents/`, then run the index command again.

## Stage 2: chunking experiment

Set `CHUNKING_STRATEGY` in `.env` to `fixed`, `fixed_overlap`, `sentence`,
`sentence_window`, `paragraph`, `section`, or `parent_child`, then index again before comparing
search results. `fixed` uses `FIXED_CHUNK_SIZE=500` lexical tokens; `fixed_overlap`
reuses `FIXED_CHUNK_OVERLAP=100` tokens from the preceding chunk. The default
production strategy is `parent_child`, which uses two-sentence child windows
with one overlapping sentence.

Run the same corpus and three answer-completeness checks through every strategy:

```bash
uv run --extra retrieval scripts/evaluate_chunking.py
```

The generated `evals/chunking-results.json` records each strategy's chunk count,
top similarity score, retrieved chunk indexes, and whether its top-three context
contains every expected answer phrase. It uses top-1 by default so an answer
that crosses a chunk boundary is visible. Read the evidence, not just the score:
cosine similarity ranks relevance but is not an accuracy percentage. The
evaluation collections (`chunking_eval_*`) are deliberately retained in Qdrant
for inspection.

`documents` is the single production collection. Users query it without choosing
a corpus or collection; every result reports its source document. Experiments
remain isolated in `chunking_eval_*` collections and are not part of normal
search.

`parent_child` treats each Markdown or numbered all-caps section as a parent and
indexes overlapping two-sentence children. Search returns the complete parent
section by default; add the surrounding child windows with `--neighbors 1`.
Use `--no-parent-context` only when the smaller child text is preferred. To
rebuild the current production handbook:

```bash
uv run --extra retrieval rag.py index scripts
uv run --extra retrieval rag.py ask "What is the retry policy for a P1 incident?" --limit 1 --neighbors 1
```

The HTTP query endpoint accepts `neighbor_count` (0–3). Its
`include_parent_context` field defaults to `true` and can be set to `false` for
child-only evidence. When multiple returned children share a parent, the
constructed context includes that parent only once; the evidence list still
shows each match and neighbor.

## What to inspect

- `chunk_text` preserves sentence boundaries.
- FastEmbed turns text into vectors.
- Qdrant ranks the question vector against chunk vectors using cosine similarity.
- The API and CLI print retrieved chunks, source, chunk number, and similarity score.

The next step will add an LLM to turn the retrieved chunks into a cited answer.

Design decisions are recorded in [`adr/`](adr/).
