import socket

import httpx
import psycopg
import redis
from fastapi import FastAPI, HTTPException, Response, status

from one_rag.answering import AnswerService
from one_rag.context import build_context
from one_rag.retrieval import RetrievalService
from one_rag.schemas import AnswerResult, DocumentIn, DocumentUpdateIn, EvaluationDocumentIn, EvaluationQueryIn, Evidence, IngestedDocument, QueryIn, QueryResult
from one_rag.settings import Settings, get_settings

app = FastAPI(title="One RAG", version="0.1.0")


def evaluation_settings(collection: str, chunking_strategy: str | None = None) -> Settings:
    updates = {"collection": collection}
    if chunking_strategy:
        updates["chunking_strategy"] = chunking_strategy
    return get_settings().model_copy(update=updates)


@app.get("/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready", tags=["health"])
def readiness(response: Response) -> dict[str, object]:
    """Reports whether every Stage 0 local dependency is reachable."""
    settings = get_settings()
    checks: dict[str, str] = {}
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2) as connection:
            connection.execute("SELECT 1")
        checks["postgres"] = "ok"
    except Exception:
        checks["postgres"] = "unavailable"
    try:
        redis.from_url(settings.redis_url, socket_connect_timeout=2).ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unavailable"
    try:
        qdrant_response = httpx.get(f"{settings.qdrant_url}/readyz", timeout=2)
        qdrant_response.raise_for_status()
        checks["qdrant"] = "ok"
    except Exception:
        checks["qdrant"] = "unavailable"
    try:
        host, port = settings.kafka_bootstrap_servers.rsplit(":", maxsplit=1)
        with socket.create_connection((host, int(port)), timeout=2):
            pass
        checks["kafka"] = "ok"
    except Exception:
        checks["kafka"] = "unavailable"
    is_ready = all(value == "ok" for value in checks.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if is_ready else "unavailable", "checks": checks}


@app.post("/v1/documents", response_model=IngestedDocument, status_code=status.HTTP_201_CREATED, tags=["rag"])
def ingest_document(document: DocumentIn) -> IngestedDocument:
    try:
        result = RetrievalService().create_document(document.source, document.text, document.tenant_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return IngestedDocument(**result)


@app.put("/v1/documents/{document_id}", response_model=IngestedDocument, tags=["rag"])
def update_document(document_id: str, document: DocumentUpdateIn) -> IngestedDocument:
    try:
        result = RetrievalService().update_document(document_id, document.source, document.text, document.tenant_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return IngestedDocument(**result)


@app.post("/v1/query", response_model=QueryResult, tags=["rag"])
def query_documents(query: QueryIn) -> QueryResult:
    evidence = [Evidence(**item) for item in RetrievalService().search(query.question, query.limit, query.neighbor_count, query.include_parent_context)]
    context = build_context(evidence)
    return QueryResult(question=query.question, context=context, evidence=evidence)


@app.post("/v1/answer", response_model=AnswerResult, tags=["rag"])
def answer_question(query: QueryIn) -> AnswerResult:
    try:
        result = AnswerService().answer(query.question, query.limit, query.neighbor_count, query.include_parent_context)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    return AnswerResult(question=query.question, **result)


@app.post("/v1/evaluations/documents", response_model=IngestedDocument, status_code=status.HTTP_201_CREATED, tags=["evaluation"])
def ingest_evaluation_document(document: EvaluationDocumentIn) -> IngestedDocument:
    settings = evaluation_settings(document.collection, document.chunking_strategy)
    result = RetrievalService(settings).create_document(document.source, document.text, document.tenant_id)
    return IngestedDocument(**result)


@app.post("/v1/evaluations/answer", response_model=AnswerResult, tags=["evaluation"])
def answer_evaluation_question(query: EvaluationQueryIn) -> AnswerResult:
    settings = evaluation_settings(query.collection)
    try:
        result = AnswerService(settings=settings, retrieval=RetrievalService(settings)).answer(query.question, query.limit, query.neighbor_count, query.include_parent_context)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    return AnswerResult(question=query.question, **result)
