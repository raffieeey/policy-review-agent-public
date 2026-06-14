from pydantic import BaseModel, Field
from typing import Literal


class PolicyMetadata(BaseModel):
    """Metadata for a policy document."""

    document_id: str
    title: str
    department: str | None = None
    jurisdiction: str | None = None
    policy_type: Literal[
        "governance", "compliance", "operational",
        "security", "hr", "financial", "other"
    ] | None = None
    source_filename: str
    content_hash: str


class PolicyChunk(BaseModel):
    """A chunk of policy document content with contextual enrichment."""

    chunk_id: str
    document_id: str
    section_id: str
    section_title: str
    text: str
    context_prefix: str
    full_text: str  # context_prefix + text
    token_count: int
    chunk_index: int
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result from hybrid retrieval (BM25 + Dense + RRF)."""

    chunk_id: str
    document_id: str
    content: str
    section_title: str
    bm25_score: float
    dense_score: float
    rrf_score: float
    metadata: dict
