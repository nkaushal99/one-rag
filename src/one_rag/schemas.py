from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=100_000)
    tenant_id: str = Field(default="local", min_length=1, max_length=200)


class DocumentUpdateIn(DocumentIn):
    pass


class IngestedDocument(BaseModel):
    document_id: str
    source: str
    chunks_indexed: int
    version: int
    content_hash: str
    embedding_reused: bool
    unchanged: bool


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=3, ge=1, le=10)
    neighbor_count: int = Field(default=0, ge=0, le=3)
    include_parent_context: bool = True


class Evidence(BaseModel):
    document_id: str
    source: str
    chunk_index: int
    text: str
    score: float
    parent_chunk_index: int | None = None
    child_chunk_index: int | None = None
    retrieval_reason: str = "match"
    parent_text: str | None = None


class QueryResult(BaseModel):
    question: str
    context: str
    evidence: list[Evidence]


class AnswerResult(QueryResult):
    answer: str
