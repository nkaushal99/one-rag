"""Index identical documents with every strategy and report retrieval evidence."""

import argparse
import json
from pathlib import Path

from one_rag.chunking import SUPPORTED_STRATEGIES
from one_rag.retrieval import RetrievalService
from one_rag.settings import Settings


def evaluate(documents_folder: Path, queries_file: Path, output: Path, limit: int) -> None:
    documents = [(path.stem, path.name, path.read_text(encoding="utf-8")) for path in sorted(documents_folder.glob("*.txt"))]
    queries = json.loads(queries_file.read_text(encoding="utf-8"))
    if not documents:
        raise ValueError("Add at least one .txt file to the documents folder.")

    report: dict[str, object] = {"limit": limit, "strategies": {}}
    for strategy in sorted(SUPPORTED_STRATEGIES):
        print(f"Evaluating {strategy}...", flush=True)
        settings = Settings(collection=f"chunking_eval_{strategy}", chunking_strategy=strategy)
        service = RetrievalService(settings)
        if service.client.collection_exists(settings.collection):
            service.client.delete_collection(settings.collection)
        chunk_count = sum(service.ingest(*document) for document in documents)
        results = []
        for query in queries:
            evidence = service.search(query["question"], limit)
            context = "\n".join(str(item["text"]) for item in evidence).lower()
            complete = all(phrase.lower() in context for phrase in query["required_phrases"])
            results.append({
                "question": query["question"],
                "complete_answer_context": complete,
                "top_score": round(float(evidence[0]["score"]), 4) if evidence else None,
                "retrieved_chunk_indexes": [item["chunk_index"] for item in evidence],
            })
        completed = sum(result["complete_answer_context"] for result in results)
        report["strategies"][strategy] = {
            "chunks_indexed": chunk_count,
            "complete_context_at_limit": f"{completed}/{len(results)}",
            "queries": results,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunking strategies on identical documents and queries.")
    parser.add_argument("--documents", type=Path, default=Path("documents"))
    parser.add_argument("--queries", type=Path, default=Path("evals/chunking-queries.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/chunking-results.json"))
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    evaluate(args.documents, args.queries, args.output, args.limit)


if __name__ == "__main__":
    main()
