"""Domain models for parsed documents, blocks, and parser exceptions."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BlockType(StrEnum):
    """Semantic type of a parsed document block."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"


class TableData(BaseModel):
    """Structured tabular representation within a document block."""

    model_config = ConfigDict(frozen=True)

    name: str | None = Field(default=None, description="Table name or sheet title")
    header: list[str] = Field(default_factory=list, description="Column header labels")
    rows: list[list[str]] = Field(default_factory=list, description="Row values as strings")


class Block(BaseModel):
    """A semantic building block of a parsed document."""

    model_config = ConfigDict(frozen=True)

    type: BlockType = Field(description="Semantic type of the block")
    text: str = Field(description="Normalized or serialized text representation")
    level: int | None = Field(default=None, description="Heading level (1-6) if block is a heading")
    page: int | None = Field(default=None, ge=1, description="1-based page number")
    section_path: list[str] = Field(
        default_factory=list, description="Active heading hierarchy at this block"
    )
    table: TableData | None = Field(
        default=None, description="Structured tabular data if block is a table"
    )


class ParsedDocument(BaseModel):
    """A fully extracted and normalized document structure."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Document title inferred or parsed from source")
    blocks: list[Block] = Field(default_factory=list, description="Ordered document blocks")
    page_count: int = Field(default=1, ge=1, description="Total number of pages")
    used_ocr: bool = Field(default=False, description="Whether OCR was triggered on any page")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary extracted metadata"
    )


class FailureReason(StrEnum):
    """Classified failure reasons for unparseable or rejected documents."""

    UNSUPPORTED = "unsupported"
    CORRUPT = "corrupt"
    ENCRYPTED = "encrypted"
    EMPTY = "empty"
    PARSE_ERROR = "parse_error"


class ParseError(Exception):
    """Exception raised when document parsing fails."""

    def __init__(self, reason: FailureReason, message: str) -> None:
        super().__init__(f"[{reason.value}] {message}")
        self.reason = reason
        self.message = message
