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
  `include_parent_context` to the retrieval pipeline, API, and CLI.
- Verification evidence: unit test confirms one child match expands to its
  sibling from the same parent only; the enterprise handbook is the live corpus.
- Before and after: before, only directly ranked text was returned; after,
  adjacent evidence and parent context can be requested explicitly.
- ADR: `adr/004-stage-2-chunking-experiments.md`.
