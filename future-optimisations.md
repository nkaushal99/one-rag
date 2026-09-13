# Planned optimisations

## OPT-004: Select retrieval context using a reproducible RAGAS baseline

- Problem: retrieval alternatives were compared manually, without an answer
  quality baseline or stable evaluator configuration.
- Planned approach: run five isolated HTTP retrieval/context configurations on
  a versioned synthetic handbook and six-answerable/two-negative golden set,
  with fixed Gemini judging and pinned BGE evaluation embeddings.
- Completion criteria: a retained local raw report, all five RAGAS metrics for
  every answerable row, deterministic negative abstention checks, and a winner
  selected by the documented score/latency rule.
