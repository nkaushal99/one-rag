import re
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from one_rag.settings import Settings


class SparseSearchUnavailable(RuntimeError):
    """OpenSearch cannot provide the required sparse half of hybrid retrieval."""


class OpenSearchSparseIndex:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenSearch(hosts=[settings.opensearch_url], timeout=5)

    @property
    def index(self) -> str:
        return self.settings.opensearch_index

    def _mapping(self) -> dict[str, object]:
        return {
            "settings": {
                "analysis": {
                    "char_filter": {
                        "terminal_punctuation": {
                            "type": "pattern_replace",
                            "pattern": "[.,;:!?]+",
                            "replacement": " ",
                        }
                    },
                    "analyzer": {
                        "whitespace_lowercase": {
                            "type": "custom",
                            "char_filter": ["terminal_punctuation"],
                            "tokenizer": "whitespace",
                            "filter": ["lowercase"],
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "qdrant_point_id": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "version": {"type": "integer"},
                    "is_active": {"type": "boolean"},
                    "chunk_index": {"type": "integer"},
                    "parent_chunk_index": {"type": "integer"},
                    "child_chunk_index": {"type": "integer"},
                    "parent_section_id": {"type": "keyword"},
                    "parent_hash": {"type": "keyword"},
                    "content_hash": {"type": "keyword"},
                    "parent_text": {"type": "text", "index": False},
                    "text": {"type": "text", "analyzer": "whitespace_lowercase"},
                }
            },
        }

    def ensure_index(self) -> None:
        try:
            if not self.client.indices.exists(index=self.index):
                self.client.indices.create(index=self.index, body=self._mapping())
        except Exception as error:
            raise SparseSearchUnavailable("OpenSearch is unavailable.") from error

    @staticmethod
    def _source(record: Any) -> dict[str, object]:
        return {**dict(record.payload), "qdrant_point_id": str(record.id)}

    @staticmethod
    def match_clause(question: str) -> dict[str, object]:
        """Use strict matching for short identifiers and broad matching for questions."""
        terms = question.split()
        identifier_like = len(terms) <= 3 and (len(terms) == 1 or any(re.search(r"[0-9_-]", term) for term in terms))
        if identifier_like:
            return {"match": {"text": {"query": question, "operator": "and"}}}
        return {"match": {"text": {"query": question, "operator": "or", "minimum_should_match": "2"}}}

    def replace_document(self, tenant_id: str, document_id: str, records: list[Any]) -> None:
        self.ensure_index()
        try:
            self.client.delete_by_query(
                index=self.index,
                body={"query": {"bool": {"filter": [{"term": {"tenant_id": tenant_id}}, {"term": {"document_id": document_id}}]}}},
                conflicts="proceed",
                refresh=True,
            )
            if records:
                actions = [{"_op_type": "index", "_index": self.index, "_id": str(record.id), "_source": self._source(record)} for record in records]
                bulk(self.client, actions, refresh=True)
        except SparseSearchUnavailable:
            raise
        except Exception as error:
            raise SparseSearchUnavailable("OpenSearch could not synchronize the document.") from error

    def rebuild(self, records: list[Any]) -> int:
        try:
            if self.client.indices.exists(index=self.index):
                self.client.indices.delete(index=self.index)
            self.client.indices.create(index=self.index, body=self._mapping())
            if records:
                actions = [{"_op_type": "index", "_index": self.index, "_id": str(record.id), "_source": self._source(record)} for record in records]
                bulk(self.client, actions, refresh=True)
            return len(records)
        except Exception as error:
            raise SparseSearchUnavailable("OpenSearch could not rebuild the sparse index.") from error

    def search(self, question: str, tenant_id: str, limit: int) -> list[dict[str, object]]:
        try:
            if not self.client.indices.exists(index=self.index):
                raise SparseSearchUnavailable("OpenSearch sparse index is missing; run scripts/reindex_opensearch.py.")
            response = self.client.search(
                index=self.index,
                body={
                    "size": limit,
                    "query": {
                        "bool": {
                            "filter": [{"term": {"tenant_id": tenant_id}}, {"term": {"is_active": True}}],
                            "must": [self.match_clause(question)],
                        }
                    },
                },
            )
            return list(response["hits"]["hits"])
        except SparseSearchUnavailable:
            raise
        except Exception as error:
            raise SparseSearchUnavailable("OpenSearch is unavailable.") from error
