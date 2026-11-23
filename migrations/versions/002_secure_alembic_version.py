"""Secure alembic_version table with RLS.

Revision ID: 002_secure_alembic_version
Revises: 001_initial_schema
Create Date: 2026-11-23 11:23:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_secure_alembic_version"
down_revision: str | None = "001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Enable RLS on alembic_version and revoke privileges from public/client roles."""
    op.execute("REVOKE ALL ON TABLE public.alembic_version FROM PUBLIC, anon, authenticated;")
    op.execute("ALTER TABLE public.alembic_version ENABLE ROW LEVEL SECURITY;")


def downgrade() -> None:
    """Disable RLS on alembic_version."""
    op.execute("ALTER TABLE public.alembic_version DISABLE ROW LEVEL SECURITY;")
