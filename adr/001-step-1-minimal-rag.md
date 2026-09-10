# ADR 001: Start with transparent retrieval

## Status

Accepted

## Context

The first milestone is to learn the RAG pipeline rather than hide it behind a framework.

## Decision

Use a Python CLI, UTF-8 text files, fixed-size overlapping chunks, local
sentence-transformer embeddings, and a local Qdrant container. Print retrieved
chunks with their source and score.

Each indexing run deletes and recreates the collection before inserting the
current chunks. This prevents deleted or edited source files from leaving stale
chunks in search results.

Do not add an API, UI, LLM, LangChain, hybrid search, reranking, metadata
filtering, or asynchronous ingestion in this milestone.

## Consequences

The pipeline is small enough to inspect end-to-end. It returns evidence rather
than a generated answer; generation and citations belong to the next milestone.
Rebuilding is inefficient for large corpora and does not preserve document
history. A later ingestion design will use stable chunk IDs and update only
changed chunks.
