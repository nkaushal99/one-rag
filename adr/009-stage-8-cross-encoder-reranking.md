# ADR 009: Evaluate opt-in cross-encoder reranking after hybrid retrieval

## Status

Accepted

## Decision

Qdrant and OpenSearch continue to provide high-recall candidates. The system
fetches up to 50 candidates from each, RRF-fuses and deduplicates them, and
optionally scores the fused top 10, 30, or 50 with FlashRank's CPU ONNX
`ms-marco-MiniLM-L-12-v2` cross-encoder. Reranking happens before selecting
primary results and before same-parent neighbor expansion.

Reranking is opt-in through `rerank_candidate_limit`; hybrid-only retrieval
remains the normal API default. Returned evidence retains RRF and channel
provenance, adding pre-rerank and reranker scores/ranks. The response trace
separates retrieval and reranking latency.

The golden RAGAS manifest compares hybrid top-5, top-10-to-5, top-30-to-5, and
top-50-to-8. The highest equal-weight RAGAS mean wins; candidates within 0.02
use lower end-to-end answer p95 latency as the tie-breaker. Evaluation Qdrant
collections receive matching isolated OpenSearch indexes.

## Consequences

The first reranked query downloads and initializes a local model, so cold-start
latency is distinct from warm query latency. FlashRank avoids a PyTorch runtime,
but reranking adds CPU work proportional to the fused candidate pool.
