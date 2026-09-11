import socket

import httpx
import psycopg
import redis
from fastapi import FastAPI, Response, status

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
