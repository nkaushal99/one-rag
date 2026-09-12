from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    source: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=100_000)


class IngestedDocument(BaseModel):
    document_id: str
    source: str
    chunks_indexed: int


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
