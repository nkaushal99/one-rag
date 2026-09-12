# ADR 005: Separate document identity from content identity

## Status

Accepted

## Decision

Every `POST /v1/documents` creates a UUID-based logical document ID. The system
uses a SHA-256 hash of normalized text as content identity, scoped to a tenant.
Identical content creates a new logical document but reuses its chunk vectors.
`PUT /v1/documents/{document_id}` is the only update operation: different
content creates the next version and deactivates earlier chunks; identical
content is an idempotent no-op.

## Consequences

Filenames cannot collide or redefine identity. Search filters to active chunks,
so old versions remain available for future audit work without appearing in
retrieval. This MVP reuses full-document vectors for identical content; stable
chunk-level hashes are a later optimisation for partially changed documents.
