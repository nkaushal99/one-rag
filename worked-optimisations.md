# Verified optimisations

## OPT-001: Measure semantic chunk boundaries

- Problem: sentence-only indexing could separate adjacent answer facts, and the
  project had no like-for-like retrieval comparison for alternative boundaries.
- Planned approach: compare fixed, overlapping fixed, sentence, sentence-window,
  paragraph, and section chunking on identical documents and questions.
- Completion criteria: configurable strategies, a repeatable evaluation, and
  retrieval evidence that records answer-context completeness.
- Completed: 2026-09-12.
- Implementation: added Stage 2 chunkers and `scripts/evaluate_chunking.py`.
- Verification evidence: unit tests cover boundaries and overlap; live results
  are recorded in `evals/chunking-results.json` after the evaluator runs.
- Before and after: before, one sentence-only strategy and no comparison; after,
  six strategies with chunks indexed, top scores, and complete-context@3.
- ADR: `adr/004-stage-2-chunking-experiments.md`.

## OPT-002: Recover context around a precise child match

- Problem: a relevant child chunk can omit the adjacent fact needed for a full
  answer, while returning an entire document adds too much unrelated context.
- Planned approach: search small child windows, retain their structural parent,
  and optionally return immediate siblings or the parent context.
- Completion criteria: no cross-section neighbors, visible match/neighbor
  reasons, and a repeatable test against parent-child chunks.
- Completed: 2026-09-12.
- Implementation: added `parent_child`, `neighbor_count`, and
  `include_parent_context` to the retrieval pipeline, API, and CLI; selected it
  as the production strategy in the unified `documents` collection.
- Verification evidence: unit test confirms one child match expands to its
  sibling from the same parent only; the enterprise handbook is the live corpus.
- Before and after: before, only directly ranked text was returned; after,
  adjacent evidence and parent context can be requested explicitly.
- ADR: `adr/004-stage-2-chunking-experiments.md`.

## OPT-003: Avoid re-embedding identical uploads

- Problem: repeated uploads and explicit updates could create duplicate vectors
  or leave stale chunks searchable.
- Planned approach: separate UUID document identity from normalized-content
  hashing, version updates, and active-chunk filtering.
- Completion criteria: exact content reuses vectors, updates increment a version,
  and inactive versions are excluded from search.
- Completed: 2026-09-13.
- Implementation: added tenant-scoped SHA-256 hashes, generated IDs, POST/PUT
  ingestion semantics, vector copying, version payloads, and active filtering.
- Verification evidence: retrieval tests cover reuse and version replacement.
- ADR: `adr/005-document-identity-and-deduplication.md`.

## OPT-004: Select retrieval context using a reproducible RAGAS baseline

- Problem: retrieval alternatives were compared manually, without an answer
  quality baseline or stable evaluator configuration.
- Planned approach: run five isolated HTTP retrieval/context configurations on
  a versioned synthetic handbook and six-answerable/two-negative golden set,
  with fixed Gemini judging and pinned BGE evaluation embeddings.
- Completion criteria: a retained local raw report, all five RAGAS metrics for
  every answerable row, deterministic negative abstention checks, and a winner
  selected by the documented score/latency rule.
- Completed: 2026-09-13.
- Implementation: added the versioned manifest and golden dataset, FastEmbed
  RAGAS adapter, paced Gemini evaluator, evaluation endpoints, runner, and
  ADR 006. The confirmed fixed judge is `gemini-3.5-flash-lite` at temperature
  zero; the evaluator embedding remains the pinned BGE revision.
- Verification evidence: `evals/golden-ragas-report.json` records all five
  metrics for all 30 answerable configuration-row combinations and a 100 percent
  abstention rate for all ten negative configuration-row combinations.
- Before and after: before, manual evidence comparison only; after, five
  isolated HTTP configurations have comparable RAGAS means. The selected
  `parent_child_top3_neighbor_parent` configuration scored 0.778 mean with
  1738.76 ms p95 latency, versus 0.158 for sentence-window top-1.
- ADR: `adr/006-reproducible-ragas-evaluation.md`.

## OPT-005: Reuse unchanged parent sections during updates

- Problem: any partial edit re-embedded every chunk in the document.
- Planned approach: persist current document and parent-section hashes in
  PostgreSQL, reuse vectors for unchanged parent sections, and embed only
  added or changed sections.
- Completion criteria: verified PostgreSQL metadata, observable reuse counts,
  and an update test that re-embeds only the changed parent section.
- Completed: 2026-09-13.
- Implementation: added `rag_documents` and `rag_document_sections`, stable
  parent-section UUIDs/hashes in Qdrant payloads, latest-only section removal,
  legacy metadata bootstrap, and inspection counters in ingest/update results.
- Verification evidence: unit tests cover change, insertion, removal, and
  duplicate-section matching. A live PostgreSQL/Qdrant update reused two Alpha
  child vectors and embedded only two changed Beta child vectors.
- Before and after: before, a single section edit embedded four children; after,
  the same two-section document embedded two and reused two.
- ADR: `adr/007-parent-section-hash-updates.md`.

## OPT-006: Fuse semantic and exact-term retrieval

- Problem: dense cosine retrieval can miss incident IDs, error names, protocol codes, and identifier-style tokens that need exact lexical matching.
- Planned approach: maintain OpenSearch BM25 documents alongside Qdrant vectors, fuse their ranked candidates with reciprocal-rank fusion, and expose both channels in returned evidence.
- Completion criteria: synchronized sparse index, visible dense/sparse traces, exact-term and semantic tests, and a live Compose dry run including sparse-index rebuild.
- Completed: 2026-09-14.
- Implementation: added the OpenSearch 2.19.1 Compose service, a versioned BM25 index with a whitespace/lowercase analyzer that strips terminal punctuation, per-document synchronization, `scripts/reindex_opensearch.py`, and RRF fusion of 20 dense plus 20 sparse candidates using `k=60`.
- Verification evidence: 34 unit tests passed; the full Compose stack reported all five dependencies ready; the sparse rebuild indexed 799 active Qdrant chunks. A live fixture returned `INC-48291`, `NullPointerException`, `PAYMENT_RETRY_V2`, `HTTP 429`, and `customer_id` as dense+sparse evidence, while the conceptual payment-retry query remained dense-led. A follow-up live P1 acknowledgement/communication question returned its communication chunk as `dense+sparse` with BM25 score `9.661691` at sparse rank 3.
- Before and after: before, only Qdrant cosine results and one similarity score were visible; after, every primary result exposes fused RRF score, dense/sparse scores and ranks, and a retrieval reason. Exact terms received sparse rank 1 in the live dry run.
- ADR: `adr/008-stage-7-hybrid-retrieval.md`.
