from unittest import TestCase

from qdrant_client import QdrantClient

from one_rag.retrieval import RetrievalService
from one_rag.metadata import InMemoryMetadataRepository
from one_rag.settings import Settings


class DeterministicEmbedder:
    def embed(self, texts: list[str]):
        for text in texts:
            normalized = text.lower()
            yield [float("credential" in normalized), float("holiday" in normalized)]


class RetrievalTests(TestCase):
    def service(self, strategy: str = "sentence") -> RetrievalService:
        service = RetrievalService.__new__(RetrievalService)
        service.settings = Settings(qdrant_url=":memory:", collection="test_documents", chunking_strategy=strategy)
        service.client = QdrantClient(":memory:")
        service.embedder = DeterministicEmbedder()
        service.metadata = InMemoryMetadataRepository()
        return service

    def test_ingest_and_query_returns_cosine_ranked_evidence(self) -> None:
        service = self.service()

        indexed = service.ingest("account-guide", "account-guide.txt", "Credentials can be changed in Account Settings. Holidays are listed elsewhere.")
        results = service.search("Where can I change credentials?", limit=1)

        self.assertEqual(indexed, 2)
        self.assertEqual(results[0]["source"], "account-guide.txt")
        self.assertEqual(results[0]["chunk_index"], 0)
        self.assertGreater(results[0]["score"], 0.9)

    def test_parent_child_search_returns_a_neighbor_from_the_same_parent(self) -> None:
        service = self.service("parent_child")
        indexed = service.ingest("guide", "guide.txt", "# Alpha\nAlpha credential detail. More alpha detail.\n\n# Beta\nBeta holiday detail. More beta detail.")
        results = service.search("credential", limit=1, neighbor_count=1, include_parent_context=True)

        self.assertEqual(indexed, 4)
        self.assertEqual([result["retrieval_reason"] for result in results], ["match", "neighbor"])
        self.assertEqual({result["parent_chunk_index"] for result in results}, {0})
        self.assertEqual({result["child_chunk_index"] for result in results}, {0, 1})
        self.assertTrue(all("Alpha" in str(result["parent_text"]) for result in results))

    def test_new_upload_gets_a_new_id_but_reuses_identical_content_embeddings(self) -> None:
        service = self.service()
        first = service.create_document("first.txt", "Credentials can be changed.", tenant_id="acme")
        second = service.create_document("copy.txt", "Credentials can be changed.", tenant_id="acme")

        self.assertNotEqual(first["document_id"], second["document_id"])
        self.assertFalse(first["embedding_reused"])
        self.assertTrue(second["embedding_reused"])
        self.assertEqual(first["content_hash"], second["content_hash"])

    def test_update_creates_a_new_version_and_hides_old_chunks_from_search(self) -> None:
        service = self.service()
        created = service.create_document("guide.txt", "Credentials can be changed.")
        updated = service.update_document(str(created["document_id"]), "guide.txt", "Holidays are listed elsewhere.")

        self.assertEqual(updated["version"], 2)
        self.assertFalse(updated["embedding_reused"])
        self.assertNotIn("Credentials", str(service.search("credential", limit=1)[0]["text"]))
        self.assertEqual(service.search("holiday", limit=1)[0]["version"], 2)

    def test_parent_child_update_reuses_unchanged_section_vectors(self) -> None:
        service = self.service("parent_child")
        created = service.create_document("guide.txt", "# Alpha\nAlpha credential detail. More alpha detail.\n\n# Beta\nBeta holiday detail. More beta detail.")
        before = service._document_records(str(created["document_id"]), "local", with_vectors=False)
        alpha_ids = {str(record.id) for record in before if record.payload["parent_chunk_index"] == 0}

        updated = service.update_document(str(created["document_id"]), "guide.txt", "# Alpha\nAlpha credential detail. More alpha detail.\n\n# Beta\nBeta changed holiday detail. More beta detail.")
        after = service._document_records(str(created["document_id"]), "local", with_vectors=False)

        self.assertEqual(updated["chunks_reused"], 2)
        self.assertEqual(updated["chunks_embedded"], 2)
        self.assertEqual(updated["sections_reused"], 1)
        self.assertEqual(alpha_ids, {str(record.id) for record in after if record.payload["parent_chunk_index"] == 0})

    def test_inserted_section_preserves_later_section_identity(self) -> None:
        service = self.service("parent_child")
        created = service.create_document("guide.txt", "# Alpha\nAlpha credential detail. More alpha detail.\n\n# Beta\nBeta holiday detail. More beta detail.")
        before = service._document_records(str(created["document_id"]), "local", with_vectors=False)
        beta_id = next(record.payload["parent_section_id"] for record in before if record.payload["parent_chunk_index"] == 1)

        updated = service.update_document(str(created["document_id"]), "guide.txt", "# New\nNew detail. More new detail.\n\n# Alpha\nAlpha credential detail. More alpha detail.\n\n# Beta\nBeta holiday detail. More beta detail.")
        after = service._document_records(str(created["document_id"]), "local", with_vectors=False)

        self.assertEqual(updated["sections_reused"], 2)
        self.assertEqual(beta_id, next(record.payload["parent_section_id"] for record in after if record.payload["parent_chunk_index"] == 2))

    def test_removed_section_deletes_only_its_children(self) -> None:
        service = self.service("parent_child")
        created = service.create_document("guide.txt", "# Alpha\nAlpha credential detail. More alpha detail.\n\n# Beta\nBeta holiday detail. More beta detail.")

        updated = service.update_document(str(created["document_id"]), "guide.txt", "# Alpha\nAlpha credential detail. More alpha detail.")
        records = service._document_records(str(created["document_id"]), "local", with_vectors=False)

        self.assertEqual(updated["chunks_reused"], 2)
        self.assertEqual(updated["chunks_deleted"], 2)
        self.assertEqual(updated["sections_deleted"], 1)
        self.assertEqual(len(records), 2)

    def test_duplicate_section_hashes_reuse_in_order(self) -> None:
        service = self.service("parent_child")
        sections = service._sections("# Same\nRepeated detail. More detail.\n\n# Same\nRepeated detail. More detail.")
        existing = service._section_rows(sections)
        updated = service._sections("# New\nNew detail. More detail.\n\n# Same\nRepeated detail. More detail.\n\n# Same\nRepeated detail. More detail.", existing)

        self.assertEqual([section["section_id"] for section in updated[1:]], [section.section_id for section in existing])
