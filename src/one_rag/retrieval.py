from collections import defaultdict, deque
from functools import lru_cache
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointIdsList, PointStruct, VectorParams

from one_rag.chunking import chunk_text, section_chunks, sentence_window_chunks
from one_rag.metadata import DocumentMetadata, PostgresMetadataRepository, SectionMetadata
from one_rag.settings import Settings, get_settings
from one_rag.sparse import OpenSearchSparseIndex


@lru_cache
def get_embedder(model_name: str) -> TextEmbedding:
    return TextEmbedding(model_name=model_name)


class RetrievalService:
    """Chunk, embed, store, and incrementally update parent-child documents."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = QdrantClient(url=self.settings.qdrant_url)
        self.embedder = get_embedder(self.settings.embedding_model)
        self.metadata = PostgresMetadataRepository(self.settings.postgres_dsn)
        self.sparse = OpenSearchSparseIndex(self.settings)

    @staticmethod
    def content_hash(text: str) -> str:
        return sha256(" ".join(text.split()).encode("utf-8")).hexdigest()

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self.embedder.embed(texts)]

    def _sections(self, text: str, existing: list[SectionMetadata] | None = None) -> list[dict[str, object]]:
        reusable: dict[str, deque[str]] = defaultdict(deque)
        for section in existing or []:
            reusable[section.section_hash].append(section.section_id)
        sections = []
        for ordinal, parent_text in enumerate(section_chunks(text)):
            parent_hash = self.content_hash(parent_text)
            section_id = reusable[parent_hash].popleft() if reusable[parent_hash] else str(uuid4())
            sections.append({"section_id": section_id, "parent_hash": parent_hash, "parent_text": parent_text, "parent_chunk_index": ordinal})
        return sections

    def _chunks(self, text: str, sections: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
        if self.settings.chunking_strategy != "parent_child":
            return [{"text": chunk, "parent_chunk_index": None, "child_chunk_index": index, "parent_text": None, "parent_section_id": None, "parent_hash": None} for index, chunk in enumerate(chunk_text(text, strategy=self.settings.chunking_strategy, chunk_size=self.settings.fixed_chunk_size, chunk_overlap=self.settings.fixed_chunk_overlap, sentence_window_size=self.settings.sentence_window_size, sentence_window_overlap=self.settings.sentence_window_overlap))]
        chunks = []
        for section in sections or self._sections(text):
            for child_index, child_text in enumerate(sentence_window_chunks(str(section["parent_text"]), self.settings.parent_child_window_size, self.settings.parent_child_window_overlap)):
                chunks.append({"text": child_text, "parent_chunk_index": section["parent_chunk_index"], "child_chunk_index": child_index, "parent_text": section["parent_text"], "parent_section_id": section["section_id"], "parent_hash": section["parent_hash"]})
        return chunks

    def _write_chunks(self, document_id: str, source: str, chunks: list[dict[str, object]], vectors: list[list[float]], tenant_id: str, content_hash: str, version: int, point_ids: list[str] | None = None) -> int:
        if not chunks:
            raise ValueError("Document text contains no indexable sentences.")
        if not self.client.collection_exists(self.settings.collection):
            self.client.create_collection(self.settings.collection, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = point_ids[index] if point_ids else str(uuid5(NAMESPACE_URL, f"{document_id}:{chunk.get('parent_section_id') or version}:{chunk['child_chunk_index']}:{index}"))
            points.append(PointStruct(id=point_id, vector=vector, payload={
                "document_id": document_id, "source": source, "tenant_id": tenant_id, "version": version, "content_hash": content_hash, "is_active": True, "chunk_index": index, "chunking_strategy": self.settings.chunking_strategy, "text": chunk["text"], "parent_chunk_index": chunk["parent_chunk_index"], "child_chunk_index": chunk["child_chunk_index"], "parent_text": chunk["parent_text"], "parent_section_id": chunk["parent_section_id"], "parent_hash": chunk["parent_hash"],
            }))
        self.client.upsert(collection_name=self.settings.collection, points=points, wait=True)
        return len(points)

    def _all_records(self, with_vectors: bool = False):
        if not self.client.collection_exists(self.settings.collection):
            return []
        records, offset = [], None
        while True:
            page, offset = self.client.scroll(collection_name=self.settings.collection, offset=offset, with_payload=True, with_vectors=with_vectors, limit=100)
            records.extend(page)
            if offset is None:
                return records

    def _document_records(self, document_id: str, tenant_id: str, with_vectors: bool = False):
        return [record for record in self._all_records(with_vectors) if record.payload.get("document_id") == document_id and record.payload.get("tenant_id", "local") == tenant_id and record.payload.get("is_active", True)]

    def _content_records(self, tenant_id: str, content_hash: str):
        return [record for record in self._all_records(with_vectors=True) if record.payload.get("tenant_id") == tenant_id and record.payload.get("content_hash") == content_hash and record.payload.get("is_active", True)]

    def _section_rows(self, sections: list[dict[str, object]]) -> list[SectionMetadata]:
        return [SectionMetadata(str(section["section_id"]), str(section["parent_hash"]), int(section["parent_chunk_index"])) for section in sections]

    def _sync_sparse_document(self, document_id: str, tenant_id: str) -> None:
        self.sparse.replace_document(tenant_id, document_id, self._document_records(document_id, tenant_id))

    def _result(self, point_id: str, payload: dict[str, object], retrieval_reason: str, score: float, include_parent_context: bool, dense_score: float | None = None, sparse_score: float | None = None, dense_rank: int | None = None, sparse_rank: int | None = None, anchor_chunk_index: int | None = None) -> dict[str, object]:
        return {"_point_id": point_id, "document_id": payload["document_id"], "source": payload["source"], "version": payload.get("version", 1), "chunk_index": payload["chunk_index"], "text": payload["text"], "score": score, "dense_score": dense_score, "sparse_score": sparse_score, "dense_rank": dense_rank, "sparse_rank": sparse_rank, "parent_chunk_index": payload.get("parent_chunk_index"), "child_chunk_index": payload.get("child_chunk_index"), "retrieval_reason": retrieval_reason, "anchor_chunk_index": anchor_chunk_index, "parent_text": payload.get("parent_text") if include_parent_context else None}

    def ingest(self, document_id: str, source: str, text: str) -> int:
        if self.client.collection_exists(self.settings.collection):
            self.client.delete(collection_name=self.settings.collection, points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]), wait=True)
        chunks = self._chunks(text)
        count = self._write_chunks(document_id, source, chunks, self._vectors([str(chunk["text"]) for chunk in chunks]), "local", self.content_hash(text), 1)
        self._sync_sparse_document(document_id, "local")
        return count

    def create_document(self, source: str, text: str, tenant_id: str = "local") -> dict[str, object]:
        content_hash, document_id = self.content_hash(text), f"doc_{uuid4().hex}"
        sections = self._sections(text) if self.settings.chunking_strategy == "parent_child" else []
        chunks = self._chunks(text, sections)
        source_records = self._content_records(tenant_id, content_hash)
        if len(source_records) == len(chunks):
            source_records.sort(key=lambda record: int(record.payload["chunk_index"]))
            vectors, reused = [list(record.vector) for record in source_records], len(chunks)
        else:
            vectors, reused = self._vectors([str(chunk["text"]) for chunk in chunks]), 0
        count = self._write_chunks(document_id, source, chunks, vectors, tenant_id, content_hash, 1)
        self.metadata.replace_current(DocumentMetadata(tenant_id, document_id, source, content_hash, 1), self._section_rows(sections))
        self._sync_sparse_document(document_id, tenant_id)
        return self._outcome(document_id, source, count, 1, content_hash, False, count - reused, reused, 0, len(sections) if not reused else 0, len(sections) if reused else 0, 0)

    def _outcome(self, document_id: str, source: str, count: int, version: int, content_hash: str, unchanged: bool, embedded: int, reused: int, deleted: int, sections_embedded: int, sections_reused: int, sections_deleted: int) -> dict[str, object]:
        return {"document_id": document_id, "source": source, "chunks_indexed": count, "version": version, "content_hash": content_hash, "embedding_reused": bool(reused), "unchanged": unchanged, "chunks_embedded": embedded, "chunks_reused": reused, "chunks_deleted": deleted, "sections_embedded": sections_embedded, "sections_reused": sections_reused, "sections_deleted": sections_deleted}

    def _bootstrap_sections(self, records) -> list[SectionMetadata]:
        parents: dict[int, tuple[str, str]] = {}
        for record in records:
            payload = record.payload
            if payload.get("parent_text") is None or payload.get("parent_chunk_index") is None:
                return []
            ordinal = int(payload["parent_chunk_index"])
            parents.setdefault(ordinal, (str(payload.get("parent_section_id") or uuid4()), self.content_hash(str(payload["parent_text"]))))
        return [SectionMetadata(section_id, section_hash, ordinal) for ordinal, (section_id, section_hash) in sorted(parents.items())]

    def _legacy_update(self, document_id: str, source: str, text: str, tenant_id: str, records, version: int, content_hash: str) -> dict[str, object]:
        self.client.delete(collection_name=self.settings.collection, points_selector=PointIdsList(points=[record.id for record in records]), wait=True)
        chunks = self._chunks(text)
        count = self._write_chunks(document_id, source, chunks, self._vectors([str(chunk["text"]) for chunk in chunks]), tenant_id, content_hash, version)
        self.metadata.replace_current(DocumentMetadata(tenant_id, document_id, source, content_hash, version), [])
        self._sync_sparse_document(document_id, tenant_id)
        return self._outcome(document_id, source, count, version, content_hash, False, count, 0, len(records), 0, 0, 0)

    def update_document(self, document_id: str, source: str, text: str, tenant_id: str = "local") -> dict[str, object]:
        records = self._document_records(document_id, tenant_id, with_vectors=True)
        if not records:
            raise ValueError("Document does not exist for this tenant.")
        stored = self.metadata.get_document(tenant_id, document_id)
        current_hash = stored.content_hash if stored else str(records[0].payload.get("content_hash", ""))
        version = stored.version if stored else max(int(record.payload.get("version", 1)) for record in records)
        content_hash = self.content_hash(text)
        if current_hash == content_hash:
            return self._outcome(document_id, source, len(records), version, content_hash, True, 0, len(records), 0, 0, len(self.metadata.sections(tenant_id, document_id)), 0)
        new_version = version + 1
        if self.settings.chunking_strategy != "parent_child":
            return self._legacy_update(document_id, source, text, tenant_id, records, new_version, content_hash)
        old_sections = self.metadata.sections(tenant_id, document_id) or self._bootstrap_sections(records)
        if not old_sections:
            return self._legacy_update(document_id, source, text, tenant_id, records, new_version, content_hash)
        sections = self._sections(text, old_sections)
        chunks = self._chunks(text, sections)
        by_section: dict[str, list] = defaultdict(list)
        for record in records:
            section_id = str(record.payload.get("parent_section_id") or next((section.section_id for section in old_sections if section.ordinal == record.payload.get("parent_chunk_index")), ""))
            by_section[section_id].append(record)
        for section_records in by_section.values():
            section_records.sort(key=lambda record: int(record.payload.get("child_chunk_index", 0)))
        old_ids, new_ids = {section.section_id for section in old_sections}, {str(section["section_id"]) for section in sections}
        reusable, changed_ids = old_ids & new_ids, old_ids - new_ids
        delete_ids = [record.id for section_id in changed_ids for record in by_section.get(section_id, [])]
        by_new_section: dict[str, list[dict[str, object]]] = defaultdict(list)
        for chunk in chunks:
            by_new_section[str(chunk["parent_section_id"])].append(chunk)
        writes, vectors, point_ids, reused_count = [], [], [], 0
        for section in sections:
            section_id, children, prior = str(section["section_id"]), by_new_section[str(section["section_id"])], by_section.get(str(section["section_id"]), [])
            if section_id in reusable and len(prior) == len(children):
                writes.extend(children); vectors.extend([list(record.vector) for record in prior]); point_ids.extend([str(record.id) for record in prior]); reused_count += len(prior)
            else:
                writes.extend(children); vectors.extend(self._vectors([str(child["text"]) for child in children])); point_ids.extend([str(uuid5(NAMESPACE_URL, f"{document_id}:{section_id}:{child['child_chunk_index']}")) for child in children]); delete_ids.extend(record.id for record in prior)
        if delete_ids:
            self.client.delete(collection_name=self.settings.collection, points_selector=PointIdsList(points=list(dict.fromkeys(delete_ids))), wait=True)
        count = self._write_chunks(document_id, source, writes, vectors, tenant_id, content_hash, new_version, point_ids)
        self.metadata.replace_current(DocumentMetadata(tenant_id, document_id, source, content_hash, new_version), self._section_rows(sections))
        self._sync_sparse_document(document_id, tenant_id)
        return self._outcome(document_id, source, count, new_version, content_hash, False, count - reused_count, reused_count, len(set(delete_ids)), len(new_ids - old_ids), len(reusable), len(changed_ids))

    def search(self, question: str, limit: int, neighbor_count: int = 1, include_parent_context: bool = True, tenant_id: str = "local") -> list[dict[str, object]]:
        if not self.client.collection_exists(self.settings.collection):
            return []
        candidate_limit = self.settings.hybrid_candidate_limit
        points = self.client.query_points(collection_name=self.settings.collection, query=self._vectors([question])[0], query_filter=Filter(must=[FieldCondition(key="is_active", match=MatchValue(value=True)), FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]), limit=candidate_limit).points
        sparse_hits = self.sparse.search(question, tenant_id, candidate_limit)
        candidates: dict[str, dict[str, object]] = {}
        for rank, point in enumerate(points, start=1):
            point_id = str(point.id)
            candidates[point_id] = {"payload": dict(point.payload), "dense_score": point.score, "dense_rank": rank, "sparse_score": None, "sparse_rank": None}
        for rank, hit in enumerate(sparse_hits, start=1):
            source = dict(hit["_source"])
            point_id = str(source["qdrant_point_id"])
            candidate = candidates.setdefault(point_id, {"payload": source, "dense_score": None, "dense_rank": None, "sparse_score": None, "sparse_rank": None})
            candidate["sparse_score"], candidate["sparse_rank"] = float(hit["_score"]), rank
        ranked = []
        for point_id, candidate in candidates.items():
            dense_rank, sparse_rank = candidate["dense_rank"], candidate["sparse_rank"]
            score = sum(1 / (self.settings.hybrid_rrf_k + rank) for rank in (dense_rank, sparse_rank) if rank is not None)
            reason = "dense+sparse" if dense_rank and sparse_rank else "dense" if dense_rank else "sparse"
            ranked.append((score, point_id, candidate, reason))
        ranked.sort(key=lambda item: item[0], reverse=True)
        results = [self._result(point_id, candidate["payload"], reason, score, include_parent_context, candidate["dense_score"], candidate["sparse_score"], candidate["dense_rank"], candidate["sparse_rank"]) for score, point_id, candidate, reason in ranked[:limit]]
        if neighbor_count < 1:
            return results
        records, selected = self._all_records(), {(item["document_id"], item["chunk_index"]) for item in results}
        for match in list(results):
            payload = match
            siblings = [record for record in records if record.payload.get("document_id") == payload["document_id"] and record.payload.get("tenant_id", "local") == tenant_id and record.payload.get("parent_chunk_index") == payload.get("parent_chunk_index") and record.payload.get("is_active", True)]
            siblings.sort(key=lambda record: int(record.payload["child_chunk_index"]))
            for record in siblings:
                distance, key = abs(int(record.payload["child_chunk_index"]) - int(payload["child_chunk_index"])), (record.payload["document_id"], record.payload["chunk_index"])
                if 0 < distance <= neighbor_count and key not in selected:
                    results.append(self._result(str(record.id), dict(record.payload), "neighbor", float(match["score"]), include_parent_context, match["dense_score"], match["sparse_score"], match["dense_rank"], match["sparse_rank"], int(match["chunk_index"]))); selected.add(key)
        return results

    def reindex_existing_collection(self) -> int:
        if not self.client.collection_exists(self.settings.collection):
            return 0
        records = [record for record in self._all_records() if record.payload.get("text")]
        if not records:
            return 0
        texts, vectors = [record.payload["text"] for record in records], self._vectors([record.payload["text"] for record in records])
        self.client.delete_collection(self.settings.collection)
        self.client.create_collection(self.settings.collection, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
        points = [PointStruct(id=str(uuid5(NAMESPACE_URL, f"{record.payload.get('document_id', 'unknown')}:{index}")), vector=vector, payload={**record.payload, "chunk_index": int(record.payload.get("chunk_index", index)), "is_active": True}) for index, (record, vector) in enumerate(zip(records, vectors))]
        self.client.upsert(collection_name=self.settings.collection, points=points, wait=True)
        self.sparse.rebuild(self._all_records())
        return len(points)
