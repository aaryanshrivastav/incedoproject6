"""Independent pure-Python access oracle for detecting retrieval/context leaks.

Intentionally implemented WITHOUT importing ``kb.retrieval.filters`` so it serves
as an independent check on SQL access filtering (see docs/DECISIONS.md D-028).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kb.db.models import ChunkModel, DocumentModel, HubDocumentModel
from kb.domain.retrieval import AccessContext, ScoredChunk


@dataclass(frozen=True)
class OracleDocRecord:
    """Minimal document + chunk metadata evaluated by the pure-Python oracle."""

    chunk_id: UUID
    chunk_is_active: bool
    doc_id: UUID
    doc_org_id: UUID
    doc_status: str
    doc_scope: str
    doc_acl_tags: tuple[str, ...]
    doc_hub_ids: frozenset[UUID]


def is_record_eligible(record: OracleDocRecord, access: AccessContext) -> bool:
    """Pure-Python access decision matching invariant I1."""
    if not record.chunk_is_active:
        return False
    if record.doc_status != "active":
        return False
    if record.doc_org_id != access.org_id:
        return False

    if record.doc_scope != "global" and access.hub_id not in record.doc_hub_ids:
        return False

    if len(record.doc_acl_tags) > 0:
        if not set(record.doc_acl_tags).intersection(access.user_tags):
            return False

    return True


async def leaks(
    access: AccessContext,
    retrieved_chunks: Sequence[ScoredChunk | UUID],
    session: AsyncSession,
    *,
    context_chunk_ids: Sequence[UUID] = (),
) -> list[UUID]:
    """Return chunk IDs present in retrieved or context chunks that the oracle deems ineligible."""
    candidate_ids: list[UUID] = []
    seen: set[UUID] = set()
    for entry in list(retrieved_chunks) + list(context_chunk_ids):
        cid = entry.chunk_id if isinstance(entry, ScoredChunk) else entry
        if cid not in seen:
            seen.add(cid)
            candidate_ids.append(cid)

    if not candidate_ids:
        return []

    chunk_stmt = (
        select(
            ChunkModel.id,
            ChunkModel.is_active,
            DocumentModel.id,
            DocumentModel.org_id,
            DocumentModel.status,
            DocumentModel.scope,
            DocumentModel.acl_tags,
        )
        .join(DocumentModel, DocumentModel.id == ChunkModel.document_id)
        .where(ChunkModel.id.in_(candidate_ids))
    )
    rows = (await session.execute(chunk_stmt)).all()
    if not rows:
        return list(candidate_ids)

    doc_ids = {UUID(str(r[2])) for r in rows}
    hub_stmt = select(HubDocumentModel.document_id, HubDocumentModel.hub_id).where(
        HubDocumentModel.document_id.in_(doc_ids)
    )
    doc_hubs_map: dict[UUID, set[UUID]] = {}
    for doc_id_raw, hub_id_raw in (await session.execute(hub_stmt)).all():
        did = UUID(str(doc_id_raw))
        hid = UUID(str(hub_id_raw))
        doc_hubs_map.setdefault(did, set()).add(hid)

    records_by_chunk: dict[UUID, OracleDocRecord] = {}
    for cid_raw, c_active, did_raw, org_raw, status, scope, acl_tags in rows:
        cid = UUID(str(cid_raw))
        did = UUID(str(did_raw))
        records_by_chunk[cid] = OracleDocRecord(
            chunk_id=cid,
            chunk_is_active=bool(c_active),
            doc_id=did,
            doc_org_id=UUID(str(org_raw)),
            doc_status=str(status),
            doc_scope=str(scope),
            doc_acl_tags=tuple(str(t) for t in (acl_tags or [])),
            doc_hub_ids=frozenset(doc_hubs_map.get(did, set())),
        )

    leaked: list[UUID] = []
    for cid in candidate_ids:
        rec = records_by_chunk.get(cid)
        if rec is None or not is_record_eligible(rec, access):
            leaked.append(cid)
    return leaked
