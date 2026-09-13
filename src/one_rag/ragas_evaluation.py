"""Small, reproducible helpers for the golden RAGAS experiment."""

import asyncio
import copy
import importlib.metadata
import re
import threading
import time
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable

from fastembed import TextEmbedding

RAGAS_METRICS = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "faithfulness": "Faithfulness",
    "answer_correctness": "Answer Accuracy",
    "answer_relevancy": "Answer Relevancy",
}
ABSTENTION = "i do not know based on the provided context."


class RequestPacer:
    """Share one request-per-minute budget between answer and judge calls."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("RAGAS requests per minute must be at least 1.")
        self.interval_seconds = 60 / requests_per_minute
        self.next_allowed = 0.0
        self.lock = threading.Lock()

    def reserve(self) -> float:
        with self.lock:
            now = time.monotonic()
            allowed = max(now, self.next_allowed)
            self.next_allowed = allowed + self.interval_seconds
        return max(0, allowed - now)

    def wait(self) -> float:
        delay = self.reserve()
        if delay:
            time.sleep(delay)
        return delay

    async def wait_async(self) -> float:
        delay = self.reserve()
        if delay:
            await asyncio.sleep(delay)
        return delay

    async def defer(self, delay_seconds: float) -> None:
        with self.lock:
            self.next_allowed = max(self.next_allowed, time.monotonic() + delay_seconds)
        await asyncio.sleep(delay_seconds)


def retry_delay_seconds(error: Exception, maximum: int) -> int:
    match = re.search(r"retry (?:in|after)\s+([0-9.]+)\s*(?:seconds?|s)", str(error), re.IGNORECASE)
    return min(maximum, max(1, int(float(match.group(1)) + 1))) if match else 1


def is_rate_limited(error: Exception) -> bool:
    text = str(error).lower()
    return "429" in text or "resourceexhausted" in text or "quota" in text


class PacedLangchainLLMWrapper:
    """Adds provider-aware pacing to RAGAS's LangChain wrapper."""

    def __init__(self, langchain_llm: Any, pacer: RequestPacer, max_retry_delay_seconds: int, log: Callable[[str], None]) -> None:
        from ragas.llms import LangchainLLMWrapper

        class Wrapper(LangchainLLMWrapper):
            async def agenerate_text(inner_self, *args: Any, **kwargs: Any):
                for attempt in range(3):
                    delay = await pacer.wait_async()
                    if delay:
                        log(f"RAGAS judge: pacing for {delay:.1f}s")
                    try:
                        return await super(Wrapper, inner_self).agenerate_text(*args, **kwargs)
                    except Exception as error:
                        if not is_rate_limited(error):
                            raise
                        retry_delay = retry_delay_seconds(error, max_retry_delay_seconds)
                        if attempt == 2:
                            raise
                        log(f"RAGAS judge: provider rate limit; waiting {retry_delay}s before retrying")
                        await pacer.defer(retry_delay)

        self.wrapper = Wrapper(langchain_llm)

    def as_ragas(self):
        return self.wrapper


def validate_manifest(manifest: dict[str, Any]) -> None:
    required = {"dataset_version", "judge", "embedding", "rate_limit", "metrics", "configurations"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Manifest is missing: {', '.join(sorted(missing))}.")
    judge = manifest["judge"]
    embedding = manifest["embedding"]
    if judge.get("model") != "gemini-3.5-flash-lite" or judge.get("temperature") != 0:
        raise ValueError("The RAGAS judge must remain gemini-3.5-flash-lite at temperature 0.")
    if embedding.get("model") != "BAAI/bge-small-en-v1.5":
        raise ValueError("The RAGAS embedding model must remain BAAI/bge-small-en-v1.5.")
    if not re.fullmatch(r"[0-9a-f]{40}", str(embedding.get("revision", ""))):
        raise ValueError("The RAGAS embedding revision must be a 40-character Git SHA.")
    if manifest["rate_limit"].get("requests_per_minute") != 12 or manifest["rate_limit"].get("max_retry_delay_seconds") != 60:
        raise ValueError("The RAGAS rate-limit policy must remain 12 RPM with a 60-second maximum retry delay.")
    if list(manifest["metrics"]) != list(RAGAS_METRICS):
        raise ValueError("The RAGAS metric order must match the fixed baseline.")
    names = [configuration["name"] for configuration in manifest["configurations"]]
    if len(names) != 5 or len(names) != len(set(names)):
        raise ValueError("The manifest must define five uniquely named configurations.")


def validate_dataset(dataset: list[dict[str, Any]]) -> None:
    answerable = [row for row in dataset if row.get("answerable")]
    negatives = [row for row in dataset if not row.get("answerable")]
    if len(answerable) != 6 or len(negatives) != 2:
        raise ValueError("The golden dataset must contain six answerable and two negative rows.")
    for row in answerable:
        if not all(row.get(field) for field in ("id", "question", "reference_answer", "reference_context")):
            raise ValueError(f"Answerable row {row.get('id', '<unknown>')} is incomplete.")
    for row in negatives:
        if not row.get("id") or not row.get("question"):
            raise ValueError("Every negative row needs an id and question.")


class FastEmbedRagasEmbeddings:
    """RAGAS-compatible adapter with FastEmbed, not sentence-transformers/PyTorch."""

    def __init__(self, model_name: str, embedder: TextEmbedding | None = None) -> None:
        from ragas.embeddings import BaseRagasEmbeddings

        self._base = BaseRagasEmbeddings
        self.model_name = model_name
        self.embedder = embedder or TextEmbedding(model_name=model_name)

    def as_ragas(self):
        owner = self
        base = self._base

        class Adapter(base):
            def __init__(self) -> None:
                super().__init__()

            def embed_query(self, text: str) -> list[float]:
                return list(next(owner.embedder.embed([text])))

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [list(vector) for vector in owner.embedder.embed(texts)]

            async def aembed_query(self, text: str) -> list[float]:
                return await asyncio.to_thread(self.embed_query, text)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                return await asyncio.to_thread(self.embed_documents, texts)

        return Adapter()


def ragas_samples(rows: list[dict[str, Any]], raw_results: list[dict[str, Any]]):
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    by_id = {result["id"]: result for result in raw_results}
    samples = []
    for row in rows:
        result = by_id[row["id"]]
        samples.append(SingleTurnSample(
            user_input=row["question"],
            response=result["answer"],
            # This is the exact expanded context sent to Gemini, including a
            # parent when the configuration requests one.
            retrieved_contexts=[result["context"]] if result["context"] else [],
            reference_contexts=[row["reference_context"]],
            reference=row["reference_answer"],
        ))
    return EvaluationDataset(samples=samples)


def evaluate_answerable(rows: list[dict[str, Any]], raw_results: list[dict[str, Any]], judge: Any, embeddings: Any) -> list[dict[str, float]]:
    from ragas import evaluate
    from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    metrics = [copy.deepcopy(metric) for metric in (context_precision, context_recall, faithfulness, answer_correctness, answer_relevancy)]
    result = evaluate(ragas_samples(rows, raw_results), metrics=metrics, llm=judge, embeddings=embeddings, run_config=RunConfig(max_workers=1, max_retries=1), raise_exceptions=True, show_progress=False)
    return map_metric_scores(result.scores)


def map_metric_scores(scores: list[dict[str, Any]]) -> list[dict[str, float]]:
    mapped = []
    for score in scores:
        if set(RAGAS_METRICS) - set(score):
            raise ValueError("RAGAS did not return every required metric.")
        mapped.append({key: float(score[key]) for key in RAGAS_METRICS})
    return mapped


def is_abstention(answer: str) -> bool:
    normalised = " ".join(answer.lower().split())
    return normalised.startswith(ABSTENTION)


def aggregate(configuration: str, rows: list[dict[str, float]], latencies_ms: list[float]) -> dict[str, float | str]:
    if not rows:
        raise ValueError("Cannot aggregate an empty RAGAS result.")
    values = {metric: fmean(row[metric] for row in rows) for metric in RAGAS_METRICS}
    ordered = sorted(latencies_ms)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.95)))
    return {"configuration": configuration, **values, "mean": fmean(values.values()), "p95_latency_ms": ordered[index]}


def select_winner(aggregates: list[dict[str, float | str]]) -> dict[str, float | str]:
    if not aggregates:
        raise ValueError("Cannot select a winner without aggregate scores.")
    best_mean = max(float(item["mean"]) for item in aggregates)
    candidates = [item for item in aggregates if best_mean - float(item["mean"]) <= 0.02]
    return min(candidates, key=lambda item: (float(item["p95_latency_ms"]), str(item["configuration"])))


def package_versions() -> dict[str, str]:
    return {package: importlib.metadata.version(package) for package in ("ragas", "fastembed", "langchain-google-genai")}
