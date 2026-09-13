# One RAG — MVP

The project begins with a deliberately small local infrastructure foundation and
a transparent retrieval API. LangChain is used only for the Gemini answer step.

## Stage 0: local infrastructure

`docker compose up --build` starts these local services:

- FastAPI at `http://localhost:8000` (interactive docs: `/docs`)
- PostgreSQL at `localhost:5432` for future document/job metadata
- Qdrant at `localhost:6333` for vectors
- OpenSearch at `localhost:9200` for BM25 keyword retrieval
- Redis at `localhost:6379` for future cache/short-lived job state
- Redpanda at `localhost:19092`, a Kafka-compatible broker for future ingestion jobs

Verify the stack with `curl http://localhost:8000/health/ready`. It checks all
five dependencies and returns HTTP 503 until they are reachable. Data volumes are
named and persist across container restarts. `docker compose down` stops services;
add `-v` only when you intentionally want to delete local data.

The `.env.template` file uses host-facing addresses for local commands. Compose
overrides those addresses inside the API container. The existing CLI reads the
host-facing `QDRANT_URL` from `.env` (normally `http://localhost:6333`). Set
`GOOGLE_API_KEY` only in your ignored local `.env`; it is never returned by the API.

## Stage 1: transparent retrieval

Plain-text documents are sentence-chunked, embedded with an open-source model,
stored in Qdrant, and indexed into OpenSearch. Each query fuses Qdrant cosine
and OpenSearch BM25 rankings with reciprocal-rank fusion (RRF), so conceptual
queries use dense retrieval while exact terms such as `INC-48291`,
`NullPointerException`, `PAYMENT_RETRY_V2`, `HTTP 429`, and `customer_id` use
sparse retrieval. The API returns the retrieved evidence and exact constructed
context. `POST /v1/answer` passes that visible context to Gemini through
LangChain and returns a cited answer.

Short identifier-style queries use strict BM25 matching (for example, `HTTP 429`), while natural-language questions use BM25 OR matching with at least two terms. This lets sparse retrieval contribute useful lexical evidence without requiring every question word or inflection to appear in a single chunk.

`POST /v1/documents` accepts `source`, `text`, and an optional `tenant_id`; it
generates a UUID-based logical `document_id`. Exact-content uploads receive a
new logical ID but reuse existing chunk vectors within the same tenant.
`PUT /v1/documents/{document_id}` creates a new version of that logical document;
an unchanged normalized-content hash is an idempotent no-op. `POST /v1/query`
accepts a `question`, optional `tenant_id`, and optional `limit` (1–10), then
returns fused evidence with dense/sparse scores and ranks, a retrieval reason,
and a context string. `score` is the RRF score, not a cosine similarity. Both
are available in the running API's `/docs`.

## Stage 8: opt-in cross-encoder reranking

Hybrid retrieval maximizes recall; optional FlashRank CPU ONNX reranking then
optimizes the order of a fused candidate pool for precision. Send
`rerank_candidate_limit` as `10`, `30`, or `50` with `/v1/query` or
`/v1/answer`; omit it to retain the current hybrid-only default. `limit` still
controls how many primary chunks are returned. Evidence preserves its RRF,
dense, and sparse trace and additionally reports pre-rerank and cross-encoder
ranks/scores. The response trace reports fused candidate count and separate
retrieval/reranking latency.

The versioned RAGAS experiment compares hybrid top-5 with top-10-to-5,
top-30-to-5, and top-50-to-8 reranking. Run it with:

```bash
uv run --extra retrieval scripts/evaluate_ragas.py --fresh
```

It needs `GOOGLE_API_KEY` in `.env`, downloads FlashRank's
`ms-marco-MiniLM-L-12-v2` model on first use, and records quality plus
end-to-end/retrieval/reranking p95 latencies in its ignored local report.

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/documents -H "Content-Type: application/json" -d '{"source":"account-guide.txt","text":"Credentials can be modified from Account Settings."}'
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question":"How can I reset my password?"}'
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question":"How can I reset my password?","limit":5,"rerank_candidate_limit":30}'
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

After upgrading an existing Qdrant collection, build its BM25 companion index
with `uv run --extra retrieval scripts/reindex_opensearch.py`. This is also the
repair command if a Qdrant write succeeds but OpenSearch is unavailable. Hybrid
query and answer requests return HTTP 503 rather than falling back silently to
dense-only retrieval when OpenSearch or this index is unavailable.

## Run it

1. Start the local stack: `docker compose up --build -d`
2. Install retrieval dependencies: `uv sync --extra retrieval`
3. Copy `.env.template` to `.env` if it is not already present.
4. Use `POST /v1/documents` to index text and `POST /v1/query` or
   `POST /v1/answer` to retrieve or answer questions.

## API development

The API service bind-mounts `./src` and runs Uvicorn with reload enabled.
After the initial `docker compose up -d --build api`, changes below `src/`
reload the API automatically; no further Compose rebuild or restart is needed.
Changes to dependencies, `pyproject.toml`, the Dockerfile, or Compose settings
still require `docker compose up -d --build api`.

The API is the only supported application interface; there is no standalone
retrieval CLI.

## Stage 2: chunking experiment

Set `CHUNKING_STRATEGY` in `.env` to `fixed`, `fixed_overlap`, `sentence`,
`sentence_window`, `paragraph`, `section`, or `parent_child`, then index again before comparing
search results. `fixed` uses `FIXED_CHUNK_SIZE=500` lexical tokens; `fixed_overlap`
reuses `FIXED_CHUNK_OVERLAP=100` tokens from the preceding chunk. The default
production strategy is `parent_child_top3_neighbor_parent`: parent-child
indexing with two-sentence child windows and one overlapping sentence, top-3
retrieval, one same-section neighbor on either side where available, and the
full parent section as LLM context.

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
indexes overlapping two-sentence children. `/v1/query` and `/v1/answer` now
default to the evaluated winner: top-3 child matches, one same-section neighbor
on either side, and deduplicated parent-section context. Override `limit`,
`neighbor_count`, or `include_parent_context` only for an explicit experiment.
For example:

```bash
curl -X POST http://127.0.0.1:8000/v1/documents -H "Content-Type: application/json" -d '{"source":"enterprise-platform-handbook.txt","text":"..."}'
curl -X POST http://127.0.0.1:8000/v1/answer -H "Content-Type: application/json" -d '{"question":"What are the P1 communication deadlines?"}'
```

The HTTP query endpoint defaults to `limit=3`, `neighbor_count=1`, and
`include_parent_context=true`. It accepts `neighbor_count` (0–3) for explicit
experiments. When multiple returned children share a parent, the constructed
context includes that parent only once; the evidence list still shows each match
and neighbor.

## Reproducible RAGAS evaluation

`evals/golden-ragas-dataset.json` has six answerable and two deliberately
unanswerable handbook questions. `evals/golden-ragas-manifest.json` fixes the
judge to `gemini-3.5-flash-lite` at temperature zero and the semantic evaluator
to FastEmbed `BAAI/bge-small-en-v1.5` at its recorded immutable revision.

With Qdrant, OpenSearch, and `GOOGLE_API_KEY` in `.env`, run all four isolated HTTP
configurations against the already-running API on port 8000:

```bash
uv run --extra retrieval scripts/evaluate_ragas.py
```

Use `--base-url http://127.0.0.1:<port>` only if the API is served elsewhere.
The runner logs collection cleanup, ingestion, each answer, RAGAS scoring,
per-configuration summary, and final report location. It uses the dedicated
`/v1/evaluations/documents` and `/v1/evaluations/answer` endpoints so each
configuration remains in its own collection without restarting the API.

The runner stays under the Gemini free-tier request limit with a shared 12 RPM
budget for answer generation and RAGAS judge calls. It writes ignored local
checkpoint state after raw answers and after each completed configuration.
Re-run the same command after a quota interruption to resume; use `--fresh` to
discard that checkpoint intentionally.

It creates `golden_eval_*` Qdrant collections and writes the ignored local
`evals/golden-ragas-report.json`. The report includes raw answers, contexts,
evidence, citations, per-row RAGAS metrics, evaluator/package versions,
end-to-end plus retrieval/reranking latency, aggregates, abstention results,
and the winner. The winner is the
highest equal-weight mean of Context Precision, Context Recall, Faithfulness,
Answer Accuracy, and Answer Relevancy; scores within 0.02 use lower p95 answer
latency as the tie-breaker. Read the raw evidence alongside the scores.

## Document identity and deduplication

For direct uploads, the API generates a new UUID-based `document_id` for every
new logical document. It hashes normalized extracted text with SHA-256. The
same content uploaded as a new document therefore keeps a distinct ID while
reusing stored chunk vectors within the same `tenant_id`. A caller that intends
to replace an existing document must use `PUT /v1/documents/{document_id}`;
changed text receives a new version and obsolete chunks are replaced, while an
unchanged upload is skipped. Filenames are source metadata, never identity.

For the production `parent_child` strategy, PostgreSQL also stores the current
hash and stable identity of every parent section. Updating one section reuses
the unchanged sections' existing Qdrant vectors, embeds only the changed or new
section children, and deletes removed sections. The response exposes
`chunks_embedded`, `chunks_reused`, `chunks_deleted`, and equivalent section
counts. This is latest-only storage: old section metadata and vectors are not
kept after an update.

## What to inspect

- `chunk_text` preserves sentence boundaries.
- FastEmbed turns text into vectors.
- Qdrant ranks the question vector against chunk vectors using cosine similarity.
- The API returns retrieved chunks, source, chunk number, and similarity score.

`/v1/answer` instructs Gemini to use only retrieved context and cite its
source/chunk labels. The response also returns that exact context and evidence
for inspection.

Design decisions are recorded in [`adr/`](adr/).
