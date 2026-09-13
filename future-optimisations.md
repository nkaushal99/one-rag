# Planned optimisations

## OPT-007: Reorder hybrid candidates with a cross-encoder

- Problem: hybrid retrieval maximizes recall but its RRF order can still place a merely related chunk above the most answer-specific evidence.
- Planned approach: score fused candidates with FlashRank's CPU ONNX cross-encoder and compare top-10-to-5, top-30-to-5, and top-50-to-8 against hybrid-only retrieval.
- Completion criteria: opt-in reranking with visible scores/ranks, isolated reproducible experiments with quality and latency, and a live model-load dry run.
