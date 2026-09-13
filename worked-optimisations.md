# Verified optimisations

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
