from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from one_rag.api import app
from one_rag.context import build_context
from one_rag.schemas import Evidence


class FakeRetrievalService:
    include_parent_context: bool | None = None
    limit: int | None = None
    neighbor_count: int | None = None

    def create_document(self, source: str, text: str, tenant_id: str) -> dict[str, object]:
        return {"document_id": "doc_generated", "source": source, "chunks_indexed": 2, "version": 1, "content_hash": "hash", "embedding_reused": False, "unchanged": False}

    def search(self, question: str, limit: int, neighbor_count: int = 0, include_parent_context: bool = False) -> list[dict[str, object]]:
        self.limit = limit
        self.neighbor_count = neighbor_count
        self.include_parent_context = include_parent_context
        return [{"document_id": "handbook", "source": "handbook.txt", "chunk_index": 0, "text": "Change credentials in Account Settings.", "score": 0.91}]


class FakeAnswerService:
    def answer(self, question: str, limit: int, neighbor_count: int, include_parent_context: bool) -> dict[str, object]:
        return {"answer": "Use Account Settings. [handbook.txt | chunk 0]", "context": "[handbook.txt | chunk 0]\nChange credentials in Account Settings.", "evidence": [{"document_id": "handbook", "source": "handbook.txt", "chunk_index": 0, "text": "Change credentials in Account Settings.", "score": 0.91}]}


class GeminiUnavailableAnswerService:
    def answer(self, question: str, limit: int, neighbor_count: int, include_parent_context: bool) -> dict[str, object]:
        raise RuntimeError("Gemini could not generate an answer with the configured model.")


class ApiTests(TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("one_rag.api.RetrievalService", FakeRetrievalService)
    def test_ingest_document_returns_chunk_count(self) -> None:
        response = self.client.post("/v1/documents", json={"source": "handbook.txt", "text": "First. Second."})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["chunks_indexed"], 2)
        self.assertEqual(response.json()["document_id"], "doc_generated")

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
        self.assertEqual(service.limit, 3)
        self.assertEqual(service.neighbor_count, 1)
        self.assertTrue(service.include_parent_context)

    def test_context_includes_an_expanded_parent_only_once(self) -> None:
        evidence = [
            Evidence(document_id="handbook", source="handbook.txt", chunk_index=10, text="First child.", score=0.9, parent_chunk_index=2, parent_text="Full parent."),
            Evidence(document_id="handbook", source="handbook.txt", chunk_index=11, text="Neighbor child.", score=0.8, parent_chunk_index=2, parent_text="Full parent.", retrieval_reason="neighbor"),
        ]
        context = build_context(evidence)
        self.assertEqual(context.count("Full parent."), 1)
        self.assertIn("parent 2", context)

    @patch("one_rag.api.AnswerService", FakeAnswerService)
    def test_answer_returns_llm_text_with_the_retrieved_context(self) -> None:
        response = self.client.post("/v1/answer", json={"question": "Where do I change credentials?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Account Settings", response.json()["answer"])
        self.assertIn("handbook.txt", response.json()["context"])

    @patch("one_rag.api.AnswerService", GeminiUnavailableAnswerService)
    def test_answer_returns_a_safe_provider_error(self) -> None:
        response = self.client.post("/v1/answer", json={"question": "Where do I change credentials?"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("Gemini", response.json()["detail"])
