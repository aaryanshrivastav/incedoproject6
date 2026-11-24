"""Integration tests for access-filtered parent expansion and end-to-end retriever."""

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kb.domain.retrieval import (
    AccessContext,
    ContextConfig,
    RerankConfig,
    ScoredChunk,
)
from kb.llm.fakes import FakeEmbedder, FakeReranker
from kb.retrieval.context import ContextAssembler
from kb.retrieval.hybrid import HybridRetriever
from kb.retrieval.rerank import RerankStage
from kb.retrieval.retriever import Retriever
from tests.integration.helpers import (
    create_test_chunk,
    create_test_document,
    create_test_hub,
    make_deterministic_embedding,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parent_expansion_authorized_substituted(
    db_session: AsyncSession,
) -> None:
    """Verify that when expand_parents=True, an eligible parent chunk replaces the child chunk."""
    org_id = uuid4()
    hub = await create_test_hub(db_session, org_id, "parent-hub")

    # Document eligible for user (empty ACL tags)
    doc = await create_test_document(db_session, org_id, title="Architecture Manual", hub_id=hub.id)

    # 1. Parent chunk
    parent_chunk = await create_test_chunk(
        db_session,
        doc.id,
        text="Full Parent Section: Microservices guide and database clustering architecture.",
        ordinal=0,
    )

    # 2. Child chunk pointing to parent
    child_chunk = await create_test_chunk(
        db_session,
        doc.id,
        text="microservices architecture snippet",
        ordinal=1,
        parent_chunk_id=parent_chunk.id,
    )
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    assembler = ContextAssembler(
        config=ContextConfig(expand_parents=True, budget_tokens=2000),
        session_factory=session_factory,
    )

    child_scored = ScoredChunk(
        chunk_id=child_chunk.id,
        document_id=doc.id,
        document_title="Architecture Manual",
        source_uri=doc.source_uri,
        text=child_chunk.text,
        ordinal=child_chunk.ordinal,
        content_hash=child_chunk.content_hash,
        parent_chunk_id=parent_chunk.id,
        fused_score=0.8,
        rerank_score=0.9,
        final_score=0.9,
    )

    ctx = AccessContext(org_id=org_id, hub_id=hub.id, user_tags=frozenset())
    bundle = await assembler.assemble([child_scored], access=ctx)

    assert len(bundle.items) == 1
    # Chunk ID must now be the parent's ID and text must be the parent's text
    assert bundle.items[0].chunk_id == parent_chunk.id
    assert bundle.items[0].text == parent_chunk.text
    # Final score preserved from child that matched query
    assert bundle.items[0].final_score == 0.9


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parent_expansion_unauthorized_not_substituted(
    db_session: AsyncSession,
) -> None:
    """Negative test: A parent chunk protected by an ACL tag the user lacks is NOT substituted."""
    org_id = uuid4()
    hub = await create_test_hub(db_session, org_id, "acl-parent-hub")

    # Document 1: Public child document
    doc_public = await create_test_document(
        db_session, org_id, title="Public Summary", acl_tags=[], hub_id=hub.id
    )

    # Document 2: Restricted parent document requiring 'secret' tag
    doc_secret = await create_test_document(
        db_session, org_id, title="Secret Parent", acl_tags=["secret"], hub_id=hub.id
    )
    parent_secret_chunk = await create_test_chunk(
        db_session,
        doc_secret.id,
        text="Highly confidential parent details.",
        ordinal=0,
    )

    # Child chunk points to the restricted parent
    child_chunk = await create_test_chunk(
        db_session,
        doc_public.id,
        text="Public snippet referencing secret",
        ordinal=0,
        parent_chunk_id=parent_secret_chunk.id,
    )
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    assembler = ContextAssembler(
        config=ContextConfig(expand_parents=True, budget_tokens=2000),
        session_factory=session_factory,
    )

    child_scored = ScoredChunk(
        chunk_id=child_chunk.id,
        document_id=doc_public.id,
        document_title="Public Summary",
        source_uri=doc_public.source_uri,
        text=child_chunk.text,
        ordinal=child_chunk.ordinal,
        content_hash=child_chunk.content_hash,
        parent_chunk_id=parent_secret_chunk.id,
        fused_score=0.8,
        rerank_score=0.8,
        final_score=0.8,
    )

    # User lacks 'secret' tag -> parent query inside SQL must not return the parent
    ctx = AccessContext(org_id=org_id, hub_id=hub.id, user_tags=frozenset(["public"]))
    bundle = await assembler.assemble([child_scored], access=ctx)

    assert len(bundle.items) == 1
    # Parent substitution MUST fail due to SQL access filter; child remains intact
    assert bundle.items[0].chunk_id == child_chunk.id
    assert bundle.items[0].text == child_chunk.text
    assert "confidential" not in bundle.items[0].text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_retriever_respects_access_filter(
    db_session: AsyncSession,
) -> None:
    """Verify Retriever enforces access filter and ranks candidates correctly."""
    org_id = uuid4()
    hub_a = await create_test_hub(db_session, org_id, "prod-hub-a")
    hub_b = await create_test_hub(db_session, org_id, "prod-hub-b")

    vec = make_deterministic_embedding("database connection pooling")

    # Document in Hub A (authorized)
    doc_a = await create_test_document(
        db_session, org_id, title="Hub A Pooling Guide", scope="hub", hub_id=hub_a.id
    )
    await create_test_chunk(
        db_session,
        doc_a.id,
        text="Database connection pooling optimization guidelines",
        embedding=vec,
    )

    # Document in Hub B (unauthorized for Hub A query)
    doc_b = await create_test_document(
        db_session, org_id, title="Hub B Pooling Guide", scope="hub", hub_id=hub_b.id
    )
    await create_test_chunk(
        db_session,
        doc_b.id,
        text="Database connection pooling secret configuration for Hub B",
        embedding=vec,
    )
    await db_session.commit()

    fake_embedder = FakeEmbedder(dimension=768)
    fake_embedder.embed_map = {"connection pooling": vec}

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    hybrid_retriever = HybridRetriever(session_factory, fake_embedder)
    rerank_stage = RerankStage(FakeReranker(), RerankConfig(enabled=True, top_n=5))

    retriever = Retriever(hybrid_retriever, rerank_stage)

    # Query with Hub A access
    ctx = AccessContext(org_id=org_id, hub_id=hub_a.id, user_tags=frozenset())
    results = await retriever.retrieve("connection pooling", ctx)

    assert len(results) == 1
    assert results[0].document_title == "Hub A Pooling Guide"
    assert results[0].rerank_score is not None
    assert results[0].final_score > 0.0
