from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..schemas.documents import PolicyChunk, PolicyMetadata
from ..config.settings import settings


class ContextualChunker:
    """
    Chunk policy documents with simple contextual prefix enrichment.

    Each chunk receives a human-readable context prefix describing
    its origin (document title and section) to improve embedding
    quality and retrieval accuracy.

    Note: For Phase 1 MVP, context prefixes are generated with a
    simple template (no LLM call). Full LLM-based contextual chunking
    is deferred to Phase 2.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def _build_context_prefix(
        self,
        document_title: str,
        section_title: str,
        policy_type: str | None,
    ) -> str:
        """Build a simple context prefix for the chunk."""
        if policy_type:
            return (
                f"From '{document_title}' ({policy_type} policy), "
                f"section '{section_title}'."
            )
        return f"From '{document_title}', section '{section_title}'."

    def chunk_document(
        self,
        content: str,
        metadata: PolicyMetadata,
        sections: list[dict],
    ) -> list[PolicyChunk]:
        """
        Chunk a document into enriched PolicyChunk objects.

        Strategy:
        1. Iterate over each detected section
        2. Split long sections with RecursiveCharacterTextSplitter
        3. Attach a context prefix to each chunk
        """
        chunks: list[PolicyChunk] = []
        chunk_index = 0

        # If no sections were detected, treat entire content as one section
        if not sections:
            sections = [
                {
                    "id": "section_0",
                    "title": metadata.title,
                    "content": content,
                    "level": 0,
                }
            ]

        for section in sections:
            section_text: str = section.get("content", "")
            if not section_text.strip():
                continue

            context_prefix = self._build_context_prefix(
                document_title=metadata.title,
                section_title=section["title"],
                policy_type=metadata.policy_type,
            )

            sub_chunks = self.splitter.split_text(section_text)

            for text in sub_chunks:
                if not text.strip():
                    continue

                full_text = f"{context_prefix}\n\n{text}"

                chunk = PolicyChunk(
                    chunk_id=f"{metadata.document_id}_chunk_{chunk_index:04d}",
                    document_id=metadata.document_id,
                    section_id=section["id"],
                    section_title=section["title"],
                    text=text,
                    context_prefix=context_prefix,
                    full_text=full_text,
                    token_count=max(1, len(text) // 4),
                    chunk_index=chunk_index,
                    metadata={
                        "title": metadata.title,
                        "policy_type": metadata.policy_type,
                        "section_level": section.get("level", 0),
                    },
                )

                chunks.append(chunk)
                chunk_index += 1

        return chunks
