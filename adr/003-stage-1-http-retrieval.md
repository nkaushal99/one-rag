# ADR 003: Expose manual retrieval through HTTP

## Status

Accepted

## Context

Stage 1 needs a minimal user-query workflow that can be inspected without
introducing an LLM provider, chain framework, queue consumer, or database schema.

## Decision

Expose `POST /v1/documents` and `POST /v1/query` in FastAPI. Documents are
split on sentence boundaries. FastEmbed produces open-source embedding vectors;
Qdrant stores and cosine-ranks them. Each result carries its source, chunk index,
score, and exact text. The query response builds an explicit context string from
the returned chunks.

The document and chunk IDs are stable, and ingestion removes prior points for
the document before upserting the latest chunks. `scripts/reindex_qdrant.py`
rebuilds all stored text payloads into this schema when migrating legacy data.

## Consequences

The MVP can serve semantic user queries without an API key and lets users see
why retrieval selected a chunk. It deliberately returns evidence/context rather
than a generated answer. Prompting, citations in prose, LLM provider selection,
and richer chunking are later stages.
