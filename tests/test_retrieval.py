from unittest import TestCase

from qdrant_client import QdrantClient

from one_rag.retrieval import RetrievalService
from one_rag.settings import Settings


class DeterministicEmbedder:
    def embed(self, texts: list[str]):
        for text in texts:
            normalized = text.lower()
            yield [float("credential" in normalized), float("holiday" in normalized)]


class RetrievalTests(TestCase):
    def test_ingest_and_query_returns_cosine_ranked_evidence(self) -> None:
        service = RetrievalService.__new__(RetrievalService)
        service.settings = Settings(qdrant_url=":memory:", collection="test_documents")
        service.client = QdrantClient(":memory:")
        service.embedder = DeterministicEmbedder()

        indexed = service.ingest("account-guide", "account-guide.txt", "Credentials can be changed in Account Settings. Holidays are listed elsewhere.")
        results = service.search("Where can I change credentials?", limit=1)

        self.assertEqual(indexed, 2)
        self.assertEqual(results[0]["source"], "account-guide.txt")
        self.assertEqual(results[0]["chunk_index"], 0)
        self.assertGreater(results[0]["score"], 0.9)
