# ADR 008: Fuse dense and sparse retrieval with reciprocal-rank fusion

## Status

Accepted

## Context

Dense embeddings retrieve conceptual matches but may not retrieve operational identifiers such as incident IDs, exception names, protocol codes, and underscore-delimited fields reliably. Those exact lexical terms need BM25 retrieval without abandoning semantic search.

## Decision

Qdrant remains the dense vector store. OpenSearch stores every active Qdrant child payload in `one_rag_chunks_v1` and performs BM25 retrieval. Its whitespace-plus-lowercase analyzer preserves hyphens and underscores inside identifiers, strips terminal sentence punctuation, and matches case-insensitively.

Every query takes the first 20 candidates from each engine for the requested tenant and fuses them with reciprocal-rank fusion using `1 / (60 + rank)`. Short identifier-style queries use strict BM25 AND matching; natural-language questions use OR matching with at least two lexical terms, so a missing inflection does not suppress an otherwise useful keyword result. The top requested fused child matches expand same-parent neighbors exactly as before. `score` is the fused score; evidence also exposes each source score/rank and whether the result was dense, sparse, both, or a neighbor. Neighbors inherit the matching child trace and identify that child in `anchor_chunk_index`.

Writes update Qdrant then replace that document's OpenSearch records. The stores do not have a distributed transaction: a failed sparse write returns a service error and `scripts/reindex_opensearch.py` repairs OpenSearch from the active Qdrant payloads.

## Consequences

The application requires OpenSearch for query and answer retrieval and returns HTTP 503 when it or its index is unavailable, rather than silently running dense-only. OpenSearch adds local runtime and storage cost, while Qdrant remains the source for sparse-index rebuilds and structural neighbor expansion.
