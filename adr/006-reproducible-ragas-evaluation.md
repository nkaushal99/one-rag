# ADR 006: Establish reproducible RAGAS evaluation

- Status: accepted
- Date: 2026-09-13

## Context

Stage 2 changed both chunk boundaries and how surrounding context is expanded.
Cosine score alone cannot establish whether a grounded answer is complete. The
synthetic enterprise-platform handbook is large enough to exercise section
boundaries, adjacent facts, legacy-policy traps, and unanswerable questions.

## Decision

Keep `evals/golden-ragas-dataset.json` (six answerable and two negative rows)
and `evals/golden-ragas-manifest.json` in version control. The manifest fixes
the Gemini judge to `gemini-2.5-flash-lite` at temperature zero and fixes the
semantic evaluator to `BAAI/bge-small-en-v1.5` at immutable Hugging Face
revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The runner also records the
resolved RAGAS, FastEmbed, and LangChain Google package versions.

The runner uses a FastEmbed-to-RAGAS adapter. This keeps evaluator embeddings
on the existing ONNX/FastEmbed path and avoids installing PyTorch merely for
RAGAS semantic metrics.

Answerable rows receive Context Precision, Context Recall, Faithfulness,
Answer Accuracy, and Answer Relevancy. Negatives are not mixed into those
semantic means: they have no factual reference answer, so they are evaluated
with the deterministic required abstention sentence instead.

Five configurations run through `POST /v1/documents` and `POST /v1/answer`,
each in an isolated `golden_eval_<configuration>` Qdrant collection:

1. sentence-window top-1 child;
2. parent-child top-1 child;
3. parent-child top-1 plus one child neighbour;
4. parent-child top-1 plus one neighbour and parent context;
5. parent-child top-3 plus one neighbour and parent context.

The winner is the highest equal-weight mean of the five RAGAS metrics. When
means are within 0.02, lower answerable-row p95 latency wins. The ignored local
JSON report retains the raw answer, prompt context, evidence, citations,
per-row metrics, latency, versions, aggregate scores, and selected winner.

## Consequences

The generated report is intentionally ignored because LLM outputs vary; the
inputs, evaluator settings, runner, and ADR remain reviewable. Re-running any
baseline must preserve the same manifest values. LLM-as-judge metrics can vary
with provider behavior and are imperfect proxies for production usefulness, so
the raw evidence and deterministic negative checks remain part of inspection.
