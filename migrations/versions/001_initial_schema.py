"""Initial knowledge assistant database schema with pgvector.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-11-23 10:40:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from kb.config import get_settings

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Note: EMBEDDING_DIM is dynamically read from application settings.
# Changing EMBEDDING_DIM in config.py requires a new migration to alter the vector dimension.
settings = get_settings()
EMBEDDING_DIM = settings.EMBEDDING_DIM if settings.EMBEDDING_DIM > 0 else 768


def upgrade() -> None:
    """Apply initial database schema with pgvector, tables, and indexes."""
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Table: hubs
    op.create_table(
        "hubs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("org_id", "name", name="uq_hubs_org_name"),
    )

    # 3. Table: documents
    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_uri", sa.String(length=1024), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default=sa.text("'active'")
        ),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("scope", sa.String(length=32), nullable=False, server_default=sa.text("'hub'")),
        sa.Column("owner_hub_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "acl_tags",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["owner_hub_id"], ["hubs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "status IN ('active', 'superseded', 'failed')",
            name="ck_documents_status",
        ),
        sa.CheckConstraint("scope IN ('hub', 'global')", name="ck_documents_scope"),
        sa.UniqueConstraint(
            "org_id", "source_uri", "version", name="uq_documents_org_source_version"
        ),
    )

    # Partial unique index for one active version per source URI
    op.create_index(
        "ix_documents_active_source",
        "documents",
        ["org_id", "source_uri"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_index(
        "ix_documents_org_status_scope",
        "documents",
        ["org_id", "status", "scope"],
    )
    op.create_index(
        "ix_documents_acl_tags",
        "documents",
        ["acl_tags"],
        postgresql_using="gin",
    )

    # 4. Table: hub_documents
    op.create_table(
        "hub_documents",
        sa.Column("hub_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["hub_id"], ["hubs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("hub_id", "document_id"),
    )
    op.create_index(
        "ix_hub_documents_document_id",
        "hub_documents",
        ["document_id"],
    )

    # 5. Table: chunks
    # Note: embedding uses VECTOR(EMBEDDING_DIM), tsv is generated stored to_tsvector
    op.execute(
        f"""
        CREATE TABLE chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ordinal INT NOT NULL,
            parent_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
            text TEXT NOT NULL,
            page INT,
            section VARCHAR(512),
            token_count INT NOT NULL,
            content_hash VARCHAR(64) NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            is_active BOOLEAN NOT NULL DEFAULT true,
            embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
            tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED NOT NULL,
            CONSTRAINT uq_chunks_document_ordinal UNIQUE (document_id, ordinal)
        );
        """
    )
    op.execute(
        """
        CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv);")
    op.execute("CREATE INDEX ix_chunks_document_active ON chunks (document_id, is_active);")
    op.execute("CREATE INDEX ix_chunks_content_hash ON chunks (content_hash);")

    # 6. Table: conversations
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("hub_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["hub_id"], ["hubs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # 7. Table: messages
    op.create_table(
        "messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("trace_id", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("role IN ('user', 'assistant', 'system')", name="ck_messages_role"),
    )

    # 8. Table: feedback
    op.create_table(
        "feedback",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
    )


def downgrade() -> None:
    """Revert initial database schema and remove extension."""
    op.drop_table("feedback")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("chunks")
    op.drop_table("hub_documents")
    op.drop_table("documents")
    op.drop_table("hubs")
    op.execute("DROP EXTENSION IF EXISTS vector CASCADE;")
