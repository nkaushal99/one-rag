FROM ghcr.io/astral-sh/uv:python3.13-bookworm
WORKDIR /app
COPY . .
RUN uv sync --frozen --no-dev
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "one_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
