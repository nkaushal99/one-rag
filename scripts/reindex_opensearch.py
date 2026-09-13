"""Rebuild the OpenSearch BM25 index from active Qdrant chunk payloads."""

from one_rag.retrieval import RetrievalService


def main() -> None:
    service = RetrievalService()
    records = [record for record in service._all_records() if record.payload.get("is_active", True) and record.payload.get("text")]
    print(f"Rebuilt {service.sparse.rebuild(records)} OpenSearch chunk documents.")


if __name__ == "__main__":
    main()
