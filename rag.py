import argparse
from pathlib import Path

from one_rag.retrieval import RetrievalService


def index(folder: str) -> None:
    service = RetrievalService()
    files = sorted(Path(folder).glob("*.txt"))
    if not files:
        raise ValueError("Add at least one .txt file to the documents folder.")
    count = sum(service.ingest(path.name, path.name, path.read_text(encoding="utf-8")) for path in files)
    print(f"Indexed {count} chunks from {len(files)} documents.")


def ask(question: str, limit: int, neighbor_count: int, include_parent_context: bool) -> None:
    for item in RetrievalService().search(question, limit, neighbor_count, include_parent_context):
        print(f"\n[{item['source']} | chunk {item['chunk_index']} | {item['retrieval_reason']} | score {item['score']:.3f}]")
        print(item["parent_text"] or item["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RAG retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index")
    index_parser.add_argument("folder", nargs="?", default="documents")
    ask_parser = commands.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--limit", type=int, default=3)
    ask_parser.add_argument("--neighbors", type=int, default=0)
    ask_parser.add_argument("--no-parent-context", action="store_false", dest="parent_context")
    ask_parser.set_defaults(parent_context=True)
    args = parser.parse_args()
    if args.command == "index":
        index(args.folder)
    else:
        ask(args.question, args.limit, args.neighbors, args.parent_context)


if __name__ == "__main__":
    main()
