"""Parser protocol for document ingestion."""

from pathlib import Path
from typing import Protocol

from kb.domain.documents import ParsedDocument


class Parser(Protocol):
    """Protocol for parsing a document from the local filesystem."""

    def parse(self, path: Path) -> ParsedDocument:
        """Parse file at path into a normalized ParsedDocument.

        Raises:
            ParseError: If parsing fails or the document is invalid.
        """
        ...
