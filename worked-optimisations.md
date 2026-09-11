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
