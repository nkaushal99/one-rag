from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from one_rag.api import app


class FakeRetrievalService:
    include_parent_context: bool | None = None

    def ingest(self, document_id: str, source: str, text: str) -> int:
        return 2

    def search(self, question: str, limit: int, neighbor_count: int = 0, include_parent_context: bool = False) -> list[dict[str, object]]:
        self.include_parent_context = include_parent_context
        return [{"document_id": "handbook", "source": "handbook.txt", "chunk_index": 0, "text": "Change credentials in Account Settings.", "score": 0.91}]


class ApiTests(TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("one_rag.api.RetrievalService", FakeRetrievalService)
    def test_ingest_document_returns_chunk_count(self) -> None:
        response = self.client.post("/v1/documents", json={"document_id": "handbook", "source": "handbook.txt", "text": "First. Second."})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["chunks_indexed"], 2)

    @patch("one_rag.api.RetrievalService", FakeRetrievalService)
    def test_query_returns_context_and_scored_evidence(self) -> None:
        response = self.client.post("/v1/query", json={"question": "How do I change credentials?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("[handbook.txt | chunk 0]", response.json()["context"])
        self.assertEqual(response.json()["evidence"][0]["score"], 0.91)

    @patch("one_rag.api.RetrievalService")
    def test_query_includes_parent_context_by_default(self, retrieval_service) -> None:
        service = FakeRetrievalService()
        retrieval_service.return_value = service
        response = self.client.post("/v1/query", json={"question": "How do I recover?"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(service.include_parent_context)
