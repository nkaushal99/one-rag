from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=100_000)
    tenant_id: str = Field(default="local", min_length=1, max_length=200)


class DocumentUpdateIn(DocumentIn):
    pass


class EvaluationDocumentIn(DocumentIn):
    collection: str = Field(pattern=r"^golden_eval_[a-z0-9_]+$")
    chunking_strategy: str = Field(min_length=1, max_length=100)


class IngestedDocument(BaseModel):
    document_id: str
    source: str
    chunks_indexed: int
    version: int
    content_hash: str
    embedding_reused: bool
    unchanged: bool
    chunks_embedded: int = 0
    chunks_reused: int = 0
    chunks_deleted: int = 0
    sections_embedded: int = 0
    sections_reused: int = 0
    sections_deleted: int = 0


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    tenant_id: str = Field(default="local", min_length=1, max_length=200)
    limit: int = Field(default=6, ge=1, le=10)
    neighbor_count: int = Field(default=1, ge=0, le=3)
    include_parent_context: bool = True


class EvaluationQueryIn(QueryIn):
    collection: str = Field(pattern=r"^golden_eval_[a-z0-9_]+$")


class Evidence(BaseModel):
    document_id: str
    source: str
    chunk_index: int
    text: str
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    parent_chunk_index: int | None = None
    child_chunk_index: int | None = None
    retrieval_reason: str = "match"
    anchor_chunk_index: int | None = None
    parent_text: str | None = None


class QueryResult(BaseModel):
    question: str
    context: str
    evidence: list[Evidence]


class AnswerResult(QueryResult):
    answer: str
