"""Run the five golden RAGAS configurations through the public HTTP API."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper

from one_rag.ragas_evaluation import (
    FastEmbedRagasEmbeddings,
    RAGAS_METRICS,
    aggregate,
    evaluate_answerable,
    is_abstention,
    package_versions,
    select_winner,
    validate_dataset,
    validate_manifest,
)
from one_rag.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_server(base_url: str, process: subprocess.Popen[str]) -> None:
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError("The evaluation API server stopped before it became available.")
        try:
            if httpx.get(f"{base_url}/health/live", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("Timed out waiting for the evaluation API server.")


def clear_collection(qdrant_url: str, collection: str) -> None:
    response = httpx.delete(f"{qdrant_url.rstrip('/')}/collections/{collection}", timeout=30)
    if response.status_code not in (200, 404):
        response.raise_for_status()


def run_http_configuration(configuration: dict, dataset: list[dict], handbook: str, port: int, settings: Settings) -> dict:
    collection = f"golden_eval_{configuration['name']}"
    clear_collection(settings.qdrant_url, collection)
    environment = os.environ | {
        "COLLECTION": collection,
        "CHUNKING_STRATEGY": configuration["chunking_strategy"],
        "QDRANT_URL": settings.qdrant_url,
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "one_rag.api:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_server(base_url, process)
        ingestion = httpx.post(f"{base_url}/v1/documents", json={"source": "rag_test_enterprise_platform_handbook.txt", "text": handbook}, timeout=180)
        ingestion.raise_for_status()
        raw_rows = []
        for row in dataset:
            started = time.perf_counter()
            response = httpx.post(f"{base_url}/v1/answer", json={
                "question": row["question"],
                "limit": configuration["limit"],
                "neighbor_count": configuration["neighbor_count"],
                "include_parent_context": configuration["include_parent_context"],
            }, timeout=180)
            response.raise_for_status()
            result = response.json()
            raw_rows.append({
                "id": row["id"],
                "question": row["question"],
                "answerable": row["answerable"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "answer": result["answer"],
                "context": result["context"],
                "evidence": result["evidence"],
                "citations": [f"{item['source']}#{item['chunk_index']}" for item in result["evidence"]],
            })
        return {"collection": collection, "ingestion": ingestion.json(), "rows": raw_rows}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(ROOT / "evals" / "golden-ragas-report.json"))
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    manifest = load_json(ROOT / "evals" / "golden-ragas-manifest.json")
    dataset = load_json(ROOT / "evals" / "golden-ragas-dataset.json")
    validate_manifest(manifest)
    validate_dataset(dataset)
    handbook = (ROOT / "scripts" / "rag_test_enterprise_platform_handbook.txt").read_text(encoding="utf-8")
    settings = Settings()
    api_key = settings.google_api_key.get_secret_value() if settings.google_api_key else None
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for the RAGAS evaluation.")
    if settings.ragas_judge_model != manifest["judge"]["model"] or settings.ragas_judge_temperature != manifest["judge"]["temperature"]:
        raise RuntimeError("The runtime RAGAS judge settings do not match the versioned manifest.")
    if settings.ragas_embedding_model != manifest["embedding"]["model"] or settings.ragas_embedding_revision != manifest["embedding"]["revision"]:
        raise RuntimeError("The runtime RAGAS embedding settings do not match the versioned manifest.")

    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=settings.ragas_judge_model, temperature=settings.ragas_judge_temperature, api_key=api_key))
    embeddings = FastEmbedRagasEmbeddings(settings.ragas_embedding_model).as_ragas()
    answerable = [row for row in dataset if row["answerable"]]
    negatives = [row for row in dataset if not row["answerable"]]
    experiment = []
    for index, configuration in enumerate(manifest["configurations"]):
        http_run = run_http_configuration(configuration, dataset, handbook, args.port + index, settings)
        answerable_results = [row for row in http_run["rows"] if row["answerable"]]
        scores = evaluate_answerable(answerable, answerable_results, judge, embeddings)
        for result, metric_scores in zip(answerable_results, scores):
            result["ragas"] = metric_scores
        negative_results = [row for row in http_run["rows"] if not row["answerable"]]
        for result in negative_results:
            result["abstention_passed"] = is_abstention(result["answer"])
        summary = aggregate(configuration["name"], scores, [row["latency_ms"] for row in answerable_results])
        summary["negative_abstention_rate"] = fmean(float(row["abstention_passed"]) for row in negative_results)
        experiment.append({"configuration": configuration, "summary": summary, **http_run})

    winner = select_winner([item["summary"] for item in experiment])
    report = {
        "manifest": manifest,
        "package_versions": package_versions(),
        "source": "scripts/rag_test_enterprise_platform_handbook.txt",
        "metric_labels": RAGAS_METRICS,
        "configurations": experiment,
        "winner": winner,
        "winner_rule": "highest equal-weight RAGAS mean; when within 0.02, lower answerable p95 latency wins",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "winner": winner}, indent=2))


if __name__ == "__main__":
    main()
