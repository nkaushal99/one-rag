from functools import lru_cache
from uuid import NAMESPACE_URL, uuid5

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from one_rag.chunking import chunk_text
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

    def ingest(self, document_id: str, source: str, text: str) -> int:
        chunks = chunk_text(
            text,
            strategy=self.settings.chunking_strategy,
            chunk_size=self.settings.fixed_chunk_size,
            chunk_overlap=self.settings.fixed_chunk_overlap,
            sentence_window_size=self.settings.sentence_window_size,
            sentence_window_overlap=self.settings.sentence_window_overlap,
        )
        if not chunks:
            raise ValueError("Document text contains no indexable sentences.")
        vectors = self._vectors(chunks)
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
                    "text": chunk,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=self.settings.collection, points=points, wait=True)
        return len(points)

    def search(self, question: str, limit: int) -> list[dict[str, object]]:
        if not self.client.collection_exists(self.settings.collection):
            return []
        query_vector = self._vectors([question])[0]
        points = self.client.query_points(
            collection_name=self.settings.collection,
            query=query_vector,
            limit=limit,
        ).points
        return [
            {"document_id": point.payload["document_id"], "source": point.payload["source"], "chunk_index": point.payload["chunk_index"], "text": point.payload["text"], "score": point.score}
            for point in points
        ]

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
