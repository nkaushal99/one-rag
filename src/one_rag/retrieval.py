from functools import lru_cache
from uuid import NAMESPACE_URL, uuid5

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from one_rag.chunking import chunk_text, section_chunks, sentence_window_chunks
from one_rag.settings import Settings, get_settings

@lru_cache
def get_embedder(model_name: str) -> TextEmbedding:
    return TextEmbedding(model_name=model_name)


class RetrievalService:
    """The explicit Stage 1 chunk -> embed -> upsert -> search pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = QdrantClient(url=self.settings.qdrant_url)
        self.embedder = get_embedder(self.settings.embedding_model)

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self.embedder.embed(texts)]

    def _chunks(self, text: str) -> list[dict[str, object]]:
        if self.settings.chunking_strategy != "parent_child":
            return [{"text": chunk, "parent_chunk_index": None, "child_chunk_index": index, "parent_text": None} for index, chunk in enumerate(chunk_text(
                text,
                strategy=self.settings.chunking_strategy,
                chunk_size=self.settings.fixed_chunk_size,
                chunk_overlap=self.settings.fixed_chunk_overlap,
                sentence_window_size=self.settings.sentence_window_size,
                sentence_window_overlap=self.settings.sentence_window_overlap,
            ))]
        chunks = []
        for parent_index, parent_text in enumerate(section_chunks(text)):
            children = sentence_window_chunks(parent_text, self.settings.parent_child_window_size, self.settings.parent_child_window_overlap)
            for child_index, child_text in enumerate(children):
                chunks.append({"text": child_text, "parent_chunk_index": parent_index, "child_chunk_index": child_index, "parent_text": parent_text})
        return chunks

    def ingest(self, document_id: str, source: str, text: str) -> int:
        chunks = self._chunks(text)
        if not chunks:
            raise ValueError("Document text contains no indexable sentences.")
        vectors = self._vectors([str(chunk["text"]) for chunk in chunks])
        if not self.client.collection_exists(self.settings.collection):
            self.client.create_collection(
                self.settings.collection,
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
            )
        self.client.delete(
            collection_name=self.settings.collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))],
            ),
            wait=True,
        )
        points = [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{document_id}:{index}")),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "source": source,
                    "chunk_index": index,
                    "chunking_strategy": self.settings.chunking_strategy,
                    "text": chunk["text"],
                    "parent_chunk_index": chunk["parent_chunk_index"],
                    "child_chunk_index": chunk["child_chunk_index"],
                    "parent_text": chunk["parent_text"],
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=self.settings.collection, points=points, wait=True)
        return len(points)

    def _result(self, point, retrieval_reason: str, score: float | None = None, include_parent_context: bool = False) -> dict[str, object]:
        return {
            "document_id": point.payload["document_id"],
            "source": point.payload["source"],
            "chunk_index": point.payload["chunk_index"],
            "text": point.payload["text"],
            "score": point.score if score is None else score,
            "parent_chunk_index": point.payload.get("parent_chunk_index"),
            "child_chunk_index": point.payload.get("child_chunk_index"),
            "retrieval_reason": retrieval_reason,
            "parent_text": point.payload.get("parent_text") if include_parent_context else None,
        }

    def _all_records(self):
        records = []
        offset = None
        while True:
            page, offset = self.client.scroll(
                collection_name=self.settings.collection,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                limit=100,
            )
            records.extend(page)
            if offset is None:
                return records

    def search(self, question: str, limit: int, neighbor_count: int = 0, include_parent_context: bool = True) -> list[dict[str, object]]:
        if not self.client.collection_exists(self.settings.collection):
            return []
        query_vector = self._vectors([question])[0]
        points = self.client.query_points(
            collection_name=self.settings.collection,
            query=query_vector,
            limit=limit,
        ).points
        results = [self._result(point, "match", include_parent_context=include_parent_context) for point in points]
        if neighbor_count < 1:
            return results

        records = self._all_records()
        selected_indexes = {(item["document_id"], item["chunk_index"]) for item in results}
        for point in points:
            payload = point.payload
            parent_index = payload.get("parent_chunk_index")
            sibling_records = [
                record for record in records
                if record.payload.get("document_id") == payload["document_id"]
                and record.payload.get("parent_chunk_index") == parent_index
            ]
            sibling_records.sort(key=lambda record: int(record.payload["child_chunk_index"]))
            current_child_index = int(payload["child_chunk_index"])
            for record in sibling_records:
                distance = abs(int(record.payload["child_chunk_index"]) - current_child_index)
                key = (record.payload["document_id"], record.payload["chunk_index"])
                if 0 < distance <= neighbor_count and key not in selected_indexes:
                    results.append(self._result(record, "neighbor", point.score, include_parent_context))
                    selected_indexes.add(key)
        return results

    def reindex_existing_collection(self) -> int:
        """Rebuild all stored text payloads into the current, complete schema."""
        if not self.client.collection_exists(self.settings.collection):
            return 0
        records = []
        offset = None
        while True:
            page, offset = self.client.scroll(
                collection_name=self.settings.collection,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                limit=100,
            )
            records.extend(record for record in page if record.payload.get("text"))
            if offset is None:
                break
        if not records:
            return 0

        texts = [record.payload["text"] for record in records]
        vectors = self._vectors(texts)
        # The caller explicitly requests a full rebuild. Text payloads are read
        # before deletion, so every indexable stored chunk is preserved.
        self.client.delete_collection(self.settings.collection)
        self.client.create_collection(
            self.settings.collection,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        points = []
        for fallback_index, (record, text, vector) in enumerate(zip(records, texts, vectors)):
            payload = record.payload
            source = str(payload.get("source", "unknown"))
            document_id = str(payload.get("document_id") or source.rsplit(".", maxsplit=1)[0])
            chunk_index = int(payload.get("chunk_index", payload.get("chunk", fallback_index)))
            points.append(
                PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"{document_id}:{chunk_index}")),
                    vector=vector,
                    payload={"document_id": document_id, "source": source, "chunk_index": chunk_index, "text": text},
                )
            )
        self.client.upsert(collection_name=self.settings.collection, points=points, wait=True)
        return len(points)
