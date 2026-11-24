"""Access-filtered parent chunk retrieval.

Enforces Invariant I1: Parent chunks must pass the exact same access control
predicates (eligible_chunk_conditions) as leaf chunks.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kb.db.models import ChunkModel, DocumentModel
from kb.domain.retrieval import AccessContext
from kb.retrieval.filters import eligible_chunk_conditions


async def get_eligible_parent_chunks(
    session: AsyncSession,
    parent_chunk_ids: list[UUID],
    access: AccessContext,
) -> dict[UUID, ChunkModel]:
    """Fetch parent chunks from the repository restricting strictly by access filters.

    Args:
        session: Active async database session.
        parent_chunk_ids: List of candidate parent chunk UUIDs.
        access: Verified AccessContext (org_id, hub_id, user_tags).

    Returns:
        Mapping from parent chunk UUID to ChunkModel for eligible parents only.
        Unauthorized or inactive parent chunks are omitted.
    """
    if not parent_chunk_ids:
        return {}

    stmt = (
        select(ChunkModel)
        .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
        .where(
            ChunkModel.id.in_(parent_chunk_ids),
            *eligible_chunk_conditions(access),
        )
    )
    result = await session.execute(stmt)
    chunks = result.scalars().all()
    return {c.id: c for c in chunks}
