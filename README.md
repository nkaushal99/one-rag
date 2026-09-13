# One RAG — MVP

The project begins with a deliberately small local infrastructure foundation and
a transparent retrieval API. LangChain is used only for the Gemini answer step.

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
host-facing `QDRANT_URL` from `.env` (normally `http://localhost:6333`). Set
`GOOGLE_API_KEY` only in your ignored local `.env`; it is never returned by the API.

## Stage 1: transparent retrieval

Plain-text documents are sentence-chunked, embedded with an open-source model,
stored in Qdrant, and retrieved by cosine similarity. The API returns the
retrieved evidence and the exact constructed context. `POST /v1/answer` passes
that visible context to Gemini through LangChain and returns a cited answer.

`POST /v1/documents` accepts `source`, `text`, and an optional `tenant_id`; it
generates a UUID-based logical `document_id`. Exact-content uploads receive a
new logical ID but reuse existing chunk vectors within the same tenant.
`PUT /v1/documents/{document_id}` creates a new version of that logical document;
an unchanged normalized-content hash is an idempotent no-op. `POST /v1/query`
accepts a `question` and optional `limit` (1–10), then returns ranked evidence,
similarity scores, and a context string. Both are available in the running API's
`/docs`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/documents -H "Content-Type: application/json" -d '{"source":"account-guide.txt","text":"Credentials can be modified from Account Settings."}'
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question":"How can I reset my password?"}'
curl -X POST http://127.0.0.1:8000/v1/answer -H "Content-Type: application/json" -d '{"question":"How can I reset my password?"}'
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
4. Use `POST /v1/documents` to index text and `POST /v1/query` or
   `POST /v1/answer` to retrieve or answer questions.

The API is the only supported application interface; there is no standalone
retrieval CLI.

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
indexes overlapping two-sentence children. `/v1/query` and `/v1/answer` return
the complete parent section by default; use `neighbor_count` to add surrounding
child windows and set `include_parent_context` to `false` for child-only
evidence. For example:

```bash
curl -X POST http://127.0.0.1:8000/v1/documents -H "Content-Type: application/json" -d '{"source":"enterprise-platform-handbook.txt","text":"..."}'
curl -X POST http://127.0.0.1:8000/v1/answer -H "Content-Type: application/json" -d '{"question":"What is the retry policy for a P1 incident?","limit":1,"neighbor_count":1}'
```

The HTTP query endpoint accepts `neighbor_count` (0–3). Its
`include_parent_context` field defaults to `true` and can be set to `false` for
child-only evidence. When multiple returned children share a parent, the
constructed context includes that parent only once; the evidence list still
shows each match and neighbor.

## Reproducible RAGAS evaluation

`evals/golden-ragas-dataset.json` has six answerable and two deliberately
unanswerable handbook questions. `evals/golden-ragas-manifest.json` fixes the
judge to `gemini-2.5-flash-lite` at temperature zero and the semantic evaluator
to FastEmbed `BAAI/bge-small-en-v1.5` at its recorded immutable revision.

With Qdrant running and `GOOGLE_API_KEY` in `.env`, run all five isolated HTTP
configurations:

```bash
uv run --extra retrieval scripts/evaluate_ragas.py
```

It creates `golden_eval_*` Qdrant collections and writes the ignored local
`evals/golden-ragas-report.json`. The report includes raw answers, contexts,
evidence, citations, per-row RAGAS metrics, evaluator/package versions,
latency, aggregates, abstention results, and the winner. The winner is the
highest equal-weight mean of Context Precision, Context Recall, Faithfulness,
Answer Accuracy, and Answer Relevancy; scores within 0.02 use lower p95 answer
latency as the tie-breaker. Read the raw evidence alongside the scores.

## Document identity and deduplication

For direct uploads, the API generates a new UUID-based `document_id` for every
new logical document. It hashes normalized extracted text with SHA-256. The
same content uploaded as a new document therefore keeps a distinct ID while
reusing stored chunk vectors within the same `tenant_id`. A caller that intends
to replace an existing document must use `PUT /v1/documents/{document_id}`;
changed text receives a new version and old chunks become inactive, while an
unchanged upload is skipped. Filenames are source metadata, never identity.

## What to inspect

- `chunk_text` preserves sentence boundaries.
- FastEmbed turns text into vectors.
- Qdrant ranks the question vector against chunk vectors using cosine similarity.
- The API returns retrieved chunks, source, chunk number, and similarity score.

`/v1/answer` instructs Gemini to use only retrieved context and cite its
source/chunk labels. The response also returns that exact context and evidence
for inspection.

Design decisions are recorded in [`adr/`](adr/).
