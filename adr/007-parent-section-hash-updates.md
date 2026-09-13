# ADR 007: Reuse unchanged parent sections during document updates

## Status

Accepted

## Decision

PostgreSQL is the canonical store for the current document hash and current
parent-section hashes. `rag_documents` stores the tenant-scoped logical
document, source, normalized content hash, and version. `rag_document_sections`
stores each current parent section's stable UUID, normalized SHA-256 hash, and
current ordinal.

For `parent_child`, an update compares normalized parent-section hashes. It
keeps the UUID and Qdrant vectors of an unchanged section, updates their payload
to the new ordinal/version, and embeds only children from added or changed
sections. Removed or changed section points and rows are deleted. This is a
latest-only policy: old vectors and section metadata are not retained.

Qdrant remains the vector store. Every child payload includes `parent_section_id`
and `parent_hash` alongside `parent_chunk_index` and `parent_text`. The index is
the current display order; the UUID is the stable section identity. Existing
parent-child documents bootstrap PostgreSQL metadata from their Qdrant parent
payload on first update; incomplete legacy payloads fall back to a full rebuild.

Other chunking strategies retain whole-document update behavior.

## Consequences

Partial parent-child edits avoid embedding unrelated sections, but still write
new Qdrant payloads for reused vectors so source, version, and ordinal remain
accurate. PostgreSQL tables are created idempotently by the metadata repository
to support both existing local volumes and clean Docker environments.
