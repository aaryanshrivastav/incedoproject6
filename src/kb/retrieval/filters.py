"""Access control filter for retrieval queries.

This module is the ONLY place in the codebase where the retrieval access predicate
is defined, enforcing Invariant I1 (Access filter in the query).

Never retrieve candidates first and filter afterwards. Every query that accesses
the knowledge repository must apply these conditions directly inside SQL.
"""

from sqlalchemy import ColumnElement, and_, cast, or_, select
from sqlalchemy.dialects.postgresql import ARRAY, VARCHAR

from kb.db.models import ChunkModel, DocumentModel, HubDocumentModel
from kb.domain.retrieval import AccessContext


def eligible_chunk_conditions(access: AccessContext) -> list[ColumnElement[bool]]:
    """Return SQLAlchemy expressions restricting chunk candidates to authorized documents.

    Enforces Invariant I1:
      A chunk is eligible only if all four conditions hold:
        1. chunks.is_active is True AND documents.status = 'active'
        2. documents.org_id = access.org_id
        3. documents.scope = 'global' OR EXISTS (hub_documents row for (hub_id, doc_id))
        4. documents.acl_tags = '{}' OR documents.acl_tags && access.user_tags

    Security provenance (Invariant I1):
      - `access.org_id` and `access.user_tags` MUST originate from the verified auth token,
        never from untrusted request bodies or LLM outputs.
      - `access.hub_id` MUST be verified as an authorized hub for the requesting user.
    """
    # 1. Active chunk and active document
    cond_active = and_(
        ChunkModel.is_active.is_(True),
        DocumentModel.status == "active",
    )

    # 2. Strict tenant isolation (org_id)
    cond_org = DocumentModel.org_id == access.org_id

    # 3. Hub scope or explicit hub membership
    hub_membership_exists = (
        select(1)
        .select_from(HubDocumentModel)
        .where(
            and_(
                HubDocumentModel.hub_id == access.hub_id,
                HubDocumentModel.document_id == DocumentModel.id,
            )
        )
        .exists()
    )
    cond_scope = or_(
        DocumentModel.scope == "global",
        hub_membership_exists,
    )

    # 4. Document ACL tags
    # An empty user_tags set must see untagged documents (acl_tags = '{}') and NO tagged ones.
    user_tags_list = sorted(access.user_tags)
    user_tags_param = cast(user_tags_list, ARRAY(VARCHAR))
    empty_tags_param = cast([], ARRAY(VARCHAR))

    cond_acl = or_(
        DocumentModel.acl_tags == empty_tags_param,
        DocumentModel.acl_tags.op("&&")(user_tags_param),
    )

    return [cond_active, cond_org, cond_scope, cond_acl]
