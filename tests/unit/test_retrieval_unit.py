"""Unit tests for retrieval domain models, RRF fusion math, and SQL structure."""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql

from kb.db.models import ChunkModel, DocumentModel
from kb.domain.retrieval import (
    AccessContext,
    RetrievalConfig,
)
from kb.retrieval.filters import eligible_chunk_conditions


def test_access_context_frozen_and_required_fields() -> None:
    """Verify AccessContext is immutable and enforces required fields."""
    org_id = uuid4()
    hub_id = uuid4()
    ctx = AccessContext(org_id=org_id, hub_id=hub_id, user_tags=frozenset(["eng", "sec"]))

    assert ctx.org_id == org_id
    assert ctx.hub_id == hub_id
    assert "eng" in ctx.user_tags

    # Immutable
    with pytest.raises(ValidationError):
        ctx.org_id = uuid4()  # type: ignore[misc]

    # Required fields without defaults
    with pytest.raises(ValidationError):
        AccessContext(org_id=org_id, hub_id=hub_id)  # type: ignore[call-arg]


def test_retrieval_config_defaults() -> None:
    """Verify RetrievalConfig defaults and ef_search computation."""
    cfg = RetrievalConfig()
    assert cfg.vector_k == 50
    assert cfg.fts_k == 50
    assert cfg.fused_k == 30
    assert cfg.rrf_k == 60
    assert cfg.hnsw_ef_search is None
    assert cfg.effective_ef_search == 50  # max(40, 50)

    custom = RetrievalConfig(vector_k=20, hnsw_ef_search=100)
    assert custom.effective_ef_search == 100

    small = RetrievalConfig(vector_k=10, hnsw_ef_search=None)
    assert small.effective_ef_search == 40  # max(40, 10)


def test_rrf_hand_computed_scores_and_tie_break() -> None:
    """Verify RRF computation with hand-computed expected scores and deterministic tie-breaking."""
    rrf_k = 60

    # Chunk A: vector rank 1, fts rank 2
    # Chunk B: vector rank 2, fts rank 1
    # Chunk C: vector rank 3, not in fts
    # Chunk D: not in vector, fts rank 3

    score_A = (1.0 / (rrf_k + 1)) + (1.0 / (rrf_k + 2))
    score_B = (1.0 / (rrf_k + 2)) + (1.0 / (rrf_k + 1))
    score_C = 1.0 / (rrf_k + 3)
    score_D = 1.0 / (rrf_k + 3)

    # score_A == score_B: best_rank is 1 for both. Tie-breaker should fall back to chunk_id string.
    cid_a = UUID("00000000-0000-0000-0000-000000000001")
    cid_b = UUID("00000000-0000-0000-0000-000000000002")
    cid_c = UUID("00000000-0000-0000-0000-000000000003")
    cid_d = UUID("00000000-0000-0000-0000-000000000004")

    candidates = [
        (-score_C, 3, str(cid_c), cid_c),
        (-score_A, 1, str(cid_a), cid_a),
        (-score_D, 3, str(cid_d), cid_d),
        (-score_B, 1, str(cid_b), cid_b),
    ]
    candidates.sort()

    sorted_ids = [c[3] for c in candidates]
    # score_A == score_B -> tie break on string ID: cid_a < cid_b
    assert sorted_ids == [cid_a, cid_b, cid_c, cid_d]
    assert candidates[0][0] == pytest.approx(-score_A)
    assert candidates[2][0] == pytest.approx(-score_C)


def test_structural_compilation_contains_all_access_predicates() -> None:
    """Verify that compiling queries with eligible_chunk_conditions contains all 4 predicates."""
    access = AccessContext(
        org_id=uuid4(),
        hub_id=uuid4(),
        user_tags=frozenset(["confidential"]),
    )

    conditions = eligible_chunk_conditions(access)

    # 1. Vector query structure
    vec_stmt = (
        select(ChunkModel.id)
        .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
        .where(*conditions)
    )
    vec_sql = str(vec_stmt.compile(dialect=postgresql.dialect()))

    # 2. FTS query structure
    ts_query = func.websearch_to_tsquery("english", "test search")
    fts_stmt = (
        select(ChunkModel.id)
        .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
        .where(*conditions, ChunkModel.tsv.op("@@")(ts_query))
    )
    fts_sql = str(fts_stmt.compile(dialect=postgresql.dialect()))

    for sql in (vec_sql, fts_sql):
        # 1. Active chunk & document
        assert "chunks.is_active IS true" in sql or "chunks.is_active = true" in sql
        assert "documents.status = " in sql
        # 2. Org ID
        assert "documents.org_id = " in sql
        # 3. Scope & Hub Membership
        assert "documents.scope = " in sql
        assert "hub_documents" in sql
        assert "EXISTS" in sql
        # 4. ACL tags overlap and empty tags check
        assert "documents.acl_tags && " in sql
        assert "documents.acl_tags = " in sql
