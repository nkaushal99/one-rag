# ADR 002: Run a small local infrastructure stack

## Status

Accepted

## Context

The MVP needs a stable local foundation before ingestion and generation are
added. The requested technology direction includes FastAPI, PostgreSQL, Qdrant,
Redis, Kafka/SQS, Docker, embeddings, LLMs, and a reranker. Implementing all
application behavior now would obscure the basic RAG learning path.

## Decision

Use Docker Compose to run FastAPI, PostgreSQL, Qdrant, Redis, and Redpanda.
Redpanda exposes the Kafka protocol, needs a single local container, and can be
replaced by a managed Kafka/SQS adapter when deployment needs require it. The
API provides liveness and dependency readiness endpoints. Persistent named
volumes protect local data from routine container restarts.

Configuration is environment-based. LLM and reranker model fields are reserved
but no provider SDK, credentials, queue consumers, schemas, or business
endpoints are created in Stage 0.

## Consequences

The project can verify its local dependencies before building ingestion. This
adds service startup cost, but avoids prematurely designing asynchronous jobs or
provider abstractions. The existing CLI remains host-facing and continues to use
the Qdrant URL in `.env`.
