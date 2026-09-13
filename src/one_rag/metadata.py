from dataclasses import dataclass
from uuid import UUID

import psycopg


@dataclass(frozen=True)
class DocumentMetadata:
    tenant_id: str
    document_id: str
    source: str
    content_hash: str
    version: int


@dataclass(frozen=True)
class SectionMetadata:
    section_id: str
    section_hash: str
    ordinal: int


class PostgresMetadataRepository:
    """Current document and parent-section identities; Qdrant stores vectors."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def _schema(self, connection) -> None:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                tenant_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                source TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (tenant_id, document_id)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS rag_document_sections (
                tenant_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                section_id UUID NOT NULL,
                section_hash TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (tenant_id, document_id, section_id),
                UNIQUE (tenant_id, document_id, ordinal),
                FOREIGN KEY (tenant_id, document_id)
                    REFERENCES rag_documents (tenant_id, document_id)
                    ON DELETE CASCADE
            )
        """)

    def get_document(self, tenant_id: str, document_id: str) -> DocumentMetadata | None:
        with psycopg.connect(self.dsn) as connection:
            self._schema(connection)
            row = connection.execute("SELECT tenant_id, document_id, source, content_hash, version FROM rag_documents WHERE tenant_id = %s AND document_id = %s", (tenant_id, document_id)).fetchone()
        return DocumentMetadata(*row) if row else None

    def sections(self, tenant_id: str, document_id: str) -> list[SectionMetadata]:
        with psycopg.connect(self.dsn) as connection:
            self._schema(connection)
            rows = connection.execute("SELECT section_id, section_hash, ordinal FROM rag_document_sections WHERE tenant_id = %s AND document_id = %s ORDER BY ordinal", (tenant_id, document_id)).fetchall()
        return [SectionMetadata(str(section_id), section_hash, ordinal) for section_id, section_hash, ordinal in rows]

    def replace_current(self, document: DocumentMetadata, sections: list[SectionMetadata]) -> None:
        with psycopg.connect(self.dsn) as connection:
            self._schema(connection)
            connection.execute("""
                INSERT INTO rag_documents (tenant_id, document_id, source, content_hash, version)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, document_id) DO UPDATE SET
                    source = EXCLUDED.source, content_hash = EXCLUDED.content_hash,
                    version = EXCLUDED.version, updated_at = now()
            """, (document.tenant_id, document.document_id, document.source, document.content_hash, document.version))
            connection.execute("DELETE FROM rag_document_sections WHERE tenant_id = %s AND document_id = %s", (document.tenant_id, document.document_id))
            if sections:
                connection.cursor().executemany("INSERT INTO rag_document_sections (tenant_id, document_id, section_id, section_hash, ordinal) VALUES (%s, %s, %s, %s, %s)", [(document.tenant_id, document.document_id, UUID(section.section_id), section.section_hash, section.ordinal) for section in sections])


class InMemoryMetadataRepository:
    """Test double with the same latest-only semantics as PostgreSQL."""

    def __init__(self) -> None:
        self.documents: dict[tuple[str, str], DocumentMetadata] = {}
        self.section_rows: dict[tuple[str, str], list[SectionMetadata]] = {}

    def get_document(self, tenant_id: str, document_id: str) -> DocumentMetadata | None:
        return self.documents.get((tenant_id, document_id))

    def sections(self, tenant_id: str, document_id: str) -> list[SectionMetadata]:
        return list(self.section_rows.get((tenant_id, document_id), []))

    def replace_current(self, document: DocumentMetadata, sections: list[SectionMetadata]) -> None:
        self.documents[(document.tenant_id, document.document_id)] = document
        self.section_rows[(document.tenant_id, document.document_id)] = list(sections)
