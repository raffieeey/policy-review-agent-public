import hashlib
from pathlib import Path

from ..schemas.documents import PolicyMetadata


class PolicyDocumentParser:
    """
    Parse policy documents using Docling for high-quality extraction.

    Supports PDF, DOCX, MD, and TXT formats. Extracts structured
    sections by heading level and exports content to Markdown.
    """

    def __init__(self) -> None:
        from docling.document_converter import DocumentConverter

        self.converter = DocumentConverter()

    def parse_document(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        original_filename: str | None = None,
    ) -> tuple[PolicyMetadata, str, list[dict]]:
        """
        Parse a document and extract structured content.

        Returns:
            - PolicyMetadata
            - Full markdown content string
            - List of section dicts with id, title, content, level
        """
        result = self.converter.convert(str(file_path))

        markdown_content = result.document.export_to_markdown()

        content_hash = hashlib.sha256(
            markdown_content.encode()
        ).hexdigest()[:16]

        sections = self._extract_sections(result.document)

        source_name = original_filename if original_filename else file_path.name

        metadata = PolicyMetadata(
            document_id=f"doc_{content_hash}",
            title=result.document.name or file_path.stem,
            source_filename=source_name,
            content_hash=content_hash,
            **(metadata_overrides or {}),
        )

        return metadata, markdown_content, sections

    def _extract_sections(self, document) -> list[dict]:
        """Extract section boundaries from the parsed document."""
        sections: list[dict] = []
        current_section: dict = {
            "id": "section_0",
            "title": "Introduction",
            "content": "",
            "level": 0,
        }

        for element, _level in document.iterate_items():
            if hasattr(element, "level") and element.level is not None:
                if current_section["content"].strip():
                    sections.append(current_section.copy())

                section_id = f"section_{len(sections)}"
                current_section = {
                    "id": section_id,
                    "title": element.text if hasattr(element, "text") else "Untitled",
                    "content": "",
                    "level": element.level,
                }
            else:
                if hasattr(element, "text"):
                    current_section["content"] += element.text + "\n"

        if current_section["content"].strip():
            sections.append(current_section)

        return sections
