"""Domain models for entities in the knowledge repository."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentScope(StrEnum):
    """Scope of document visibility."""

    HUB = "hub"
    GLOBAL = "global"


class DocumentStatus(StrEnum):
    """Lifecycle status of a document."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class Hub(BaseModel):
    """Knowledge hub entity representing an isolated context or project."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    org_id: UUID
    name: str
    created_at: datetime


class Document(BaseModel):
    """Document entity representing an ingested source file or object."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    org_id: UUID
    source_uri: str
    title: str
    content_hash: str
    version: int
    status: DocumentStatus
    scope: DocumentScope
    acl_tags: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    owner_hub_id: UUID | None = None
    created_at: datetime
    updated_at: datetime


class NewChunk(BaseModel):
    """Chunk specification prepared for insertion into the repository."""

    model_config = ConfigDict(frozen=True)

    ordinal: int = Field(ge=0, description="0-based sequence ordinal within document")
    text: str = Field(description="Normalized text content of chunk")
    page: int | None = Field(default=None, ge=1, description="1-based page number")
    section: str | None = Field(default=None, description="Section heading or table context")
    token_count: int = Field(ge=0, description="Approximate token count")
    content_hash: str = Field(description="SHA-256 hash of normalized text")
    parent_chunk_id: UUID | None = Field(default=None, description="Optional parent chunk ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary chunk metadata")
    embedding: list[float] = Field(description="Dense float vector embedding")
