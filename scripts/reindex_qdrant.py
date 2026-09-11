from one_rag.retrieval import RetrievalService


def main() -> None:
    count = RetrievalService().reindex_existing_collection()
    print(f"Reindexed {count} stored chunks into the current payload schema.")


if __name__ == "__main__":
    main()
