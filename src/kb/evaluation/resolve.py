"""Resolve golden item source URIs to database document IDs and relevant chunk IDs."""

from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kb.db.models import ChunkModel, DocumentModel
from kb.domain.evaluation import GoldenItem
from kb.evaluation.golden import span_in_text


class ResolvedItem(BaseModel):
    """Runtime-resolved document and relevant chunk IDs for a GoldenItem."""

    item_id: str
    document_ids: list[UUID] = Field(default_factory=list)
    relevant_chunk_ids: set[UUID] = Field(default_factory=set)
    chunk_texts: dict[UUID, str] = Field(default_factory=dict)


async def resolve_item(
    session: AsyncSession,
    org_id: UUID,
    item: GoldenItem,
) -> ResolvedItem:
    """Resolve source_uris to active document IDs and all chunks containing any evidence span.

    Scans all active chunks of the item's source documents regardless of access context.
    """
    if not item.source_uris:
        return ResolvedItem(item_id=item.id)

    doc_stmt = select(DocumentModel.id).where(
        DocumentModel.org_id == org_id,
        DocumentModel.source_uri.in_(item.source_uris),
        DocumentModel.status == "active",
    )
    doc_ids = [UUID(str(row[0])) for row in (await session.execute(doc_stmt)).all()]
    if not doc_ids:
        return ResolvedItem(item_id=item.id)

    chunk_stmt = (
        select(ChunkModel.id, ChunkModel.text)
        .where(
            ChunkModel.document_id.in_(doc_ids),
            ChunkModel.is_active.is_(True),
        )
        .order_by(ChunkModel.ordinal)
    )
    relevant_ids: set[UUID] = set()
    chunk_texts: dict[UUID, str] = {}

    for row in (await session.execute(chunk_stmt)).all():
        cid = UUID(str(row[0]))
        text = str(row[1])
        chunk_texts[cid] = text
        if any(span_in_text(span, text) for span in item.evidence_spans):
            relevant_ids.add(cid)

    return ResolvedItem(
        item_id=item.id,
        document_ids=doc_ids,
        relevant_chunk_ids=relevant_ids,
        chunk_texts=chunk_texts,
    )
