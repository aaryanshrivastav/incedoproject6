"""SQLAlchemy 2.0 ORM models for PostgreSQL knowledge repository."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    Computed,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from kb.config import get_settings


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class HubModel(Base):
    """Knowledge hub entity representing an isolated context or project."""

    __tablename__ = "hubs"
    __table_args__ = (UniqueConstraint("org_id", "name", name="uq_hubs_org_name"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default=sa_text("gen_random_uuid()"))
    org_id: Mapped[UUID] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=sa_text("now()"), nullable=False)

    documents: Mapped[list["DocumentModel"]] = relationship(
        "DocumentModel",
        secondary="hub_documents",
        back_populates="hubs",
    )
    conversations: Mapped[list["ConversationModel"]] = relationship(
        "ConversationModel",
        back_populates="hub",
        cascade="all, delete-orphan",
    )


class DocumentModel(Base):
    """Document metadata, versioning, status, and hub scope."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'superseded', 'failed')", name="ck_documents_status"),
        CheckConstraint("scope IN ('hub', 'global')", name="ck_documents_scope"),
        UniqueConstraint("org_id", "source_uri", "version", name="uq_documents_org_source_version"),
        Index(
            "ix_documents_active_source",
            "org_id",
            "source_uri",
            unique=True,
            postgresql_where=sa_text("status = 'active'"),
        ),
        Index("ix_documents_org_status_scope", "org_id", "status", "scope"),
        Index("ix_documents_acl_tags", "acl_tags", postgresql_using="gin"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default=sa_text("gen_random_uuid()"))
    org_id: Mapped[UUID] = mapped_column(nullable=False)
    source_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope: Mapped[str] = mapped_column(String(32), nullable=False, default="hub")
    owner_hub_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("hubs.id", ondelete="SET NULL"),
        nullable=True,
    )
    acl_tags: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        server_default=sa_text("'{}'"),
    )
    created_at: Mapped[datetime] = mapped_column(server_default=sa_text("now()"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sa_text("now()"),
        onupdate=sa_text("now()"),
        nullable=False,
    )

    hubs: Mapped[list[HubModel]] = relationship(
        "HubModel",
        secondary="hub_documents",
        back_populates="documents",
    )
    chunks: Mapped[list["ChunkModel"]] = relationship(
        "ChunkModel",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class HubDocumentModel(Base):
    """Many-to-many junction between hubs and documents."""

    __tablename__ = "hub_documents"
    __table_args__ = (Index("ix_hub_documents_document_id", "document_id"),)

    hub_id: Mapped[UUID] = mapped_column(
        ForeignKey("hubs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )


# Read embedding dimension dynamically from Settings (e.g. 768 for BAAI/bge-base-en-v1.5)
_EMBEDDING_DIM = get_settings().EMBEDDING_DIM or 768


class ChunkModel(Base):
    """Text chunk with embedding vector, full-text tsvector, and provenance."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_chunks_document_active", "document_id", "is_active"),
        Index("ix_chunks_content_hash", "content_hash"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default=sa_text("gen_random_uuid()"))
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_chunk_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=sa_text("'{}'::jsonb"),
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(_EMBEDDING_DIM),
        nullable=False,
    )
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', text)", persisted=True),
        nullable=False,
    )

    document: Mapped[DocumentModel] = relationship("DocumentModel", back_populates="chunks")


class ConversationModel(Base):
    """Conversation history grouping user and assistant interaction turns."""

    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default=sa_text("gen_random_uuid()"))
    hub_id: Mapped[UUID] = mapped_column(ForeignKey("hubs.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=sa_text("now()"), nullable=False)

    hub: Mapped[HubModel] = relationship("HubModel", back_populates="conversations")
    messages: Mapped[list["MessageModel"]] = relationship(
        "MessageModel",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class MessageModel(Base):
    """Single message in a conversation thread with citations and trace metadata."""

    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="ck_messages_role"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default=sa_text("gen_random_uuid()"))
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(JSONB, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=sa_text("now()"), nullable=False)

    conversation: Mapped[ConversationModel] = relationship(
        "ConversationModel",
        back_populates="messages",
    )
    feedback: Mapped[list["FeedbackModel"]] = relationship(
        "FeedbackModel",
        back_populates="message",
        cascade="all, delete-orphan",
    )


class FeedbackModel(Base):
    """User feedback rating and optional commentary for a generated response."""

    __tablename__ = "feedback"
    __table_args__ = (CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default=sa_text("gen_random_uuid()"))
    message_id: Mapped[UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=sa_text("now()"), nullable=False)

    message: Mapped[MessageModel] = relationship("MessageModel", back_populates="feedback")
