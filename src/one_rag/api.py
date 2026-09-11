import socket

import httpx
import psycopg
import redis
from fastapi import FastAPI, HTTPException, Response, status

from one_rag.retrieval import RetrievalService
from one_rag.schemas import DocumentIn, Evidence, IngestedDocument, QueryIn, QueryResult
from one_rag.settings import get_settings

app = FastAPI(title="One RAG", version="0.1.0")


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
        chunks_indexed = RetrievalService().ingest(document.document_id, document.source, document.text)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return IngestedDocument(document_id=document.document_id, source=document.source, chunks_indexed=chunks_indexed)


@app.post("/v1/query", response_model=QueryResult, tags=["rag"])
def query_documents(query: QueryIn) -> QueryResult:
    evidence = [Evidence(**item) for item in RetrievalService().search(query.question, query.limit, query.neighbor_count, query.include_parent_context)]
    context = "\n\n".join(f"[{item.source} | chunk {item.chunk_index}]\n{item.parent_text or item.text}" for item in evidence)
    return QueryResult(question=query.question, context=context, evidence=evidence)
