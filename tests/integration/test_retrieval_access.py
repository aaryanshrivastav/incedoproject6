"""Negative-first integration tests for SQL access filtering (Invariant I1)."""

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kb.db.repositories import DocumentRepository
from kb.domain.retrieval import AccessContext, RetrievalConfig
from kb.llm.fakes import FakeEmbedder
from kb.retrieval.hybrid import HybridRetriever
from tests.integration.helpers import (
    create_test_chunk,
    create_test_document,
    create_test_hub,
    make_deterministic_embedding,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_lacking_tag_never_receives_tagged_chunk(
    db_session: AsyncSession,
) -> None:
    """Verify that a user lacking an ACL tag never receives a chunk through vector or FTS."""
    org_id = uuid4()
    hub = await create_test_hub(db_session, org_id, "finance-hub")

    # Target document tagged "finance"
    finance_text = "Executive quarterly financial statement and profit margins disclosure"
    doc_finance = await create_test_document(
        db_session,
        org_id=org_id,
        title="Q3 Financials",
        source_uri="file:///finance.pdf",
        acl_tags=["finance"],
        hub_id=hub.id,
    )
    # Give it a known vector matching query
    finance_vec = make_deterministic_embedding("financial statement")
    await create_test_chunk(
        db_session,
        document_id=doc_finance.id,
        text=finance_text,
        embedding=finance_vec,
    )
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    fake_embedder = FakeEmbedder(dimension=768)
    fake_embedder.embed_map = {"financial statement": finance_vec}
    retriever = HybridRetriever(session_factory, fake_embedder)

    # 1. User without 'finance' tag -> MUST return 0 chunks
    unauthorized_ctx = AccessContext(
        org_id=org_id, hub_id=hub.id, user_tags=frozenset(["engineering"])
    )
    results = await retriever.retrieve("financial statement", unauthorized_ctx)
    assert len(results) == 0

    # 2. User with 'finance' tag -> MUST return the chunk
    authorized_ctx = AccessContext(org_id=org_id, hub_id=hub.id, user_tags=frozenset(["finance"]))
    auth_results = await retriever.retrieve("financial statement", authorized_ctx)
    assert len(auth_results) == 1
    assert auth_results[0].document_title == "Q3 Financials"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_empty_user_tags_sees_untagged_documents_only(
    db_session: AsyncSession,
) -> None:
    """Verify that a user with empty tags sees untagged documents and no tagged ones."""
    org_id = uuid4()
    hub = await create_test_hub(db_session, org_id, "general-hub")

    doc_public = await create_test_document(
        db_session, org_id, title="Public Policy", acl_tags=[], hub_id=hub.id
    )
    await create_test_chunk(
        db_session, doc_public.id, text="Public company holiday schedule policy"
    )

    doc_restricted = await create_test_document(
        db_session, org_id, title="Restricted Bonus", acl_tags=["exec"], hub_id=hub.id
    )
    await create_test_chunk(
        db_session, doc_restricted.id, text="Restricted bonus allocation formula"
    )
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    retriever = HybridRetriever(session_factory, FakeEmbedder(dimension=768))

    ctx = AccessContext(org_id=org_id, hub_id=hub.id, user_tags=frozenset())
    results = await retriever.retrieve("policy bonus schedule", ctx)

    result_titles = [r.document_title for r in results]
    assert "Public Policy" in result_titles
    assert "Restricted Bonus" not in result_titles


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hub_isolation_never_returns_other_hub_documents(
    db_session: AsyncSession,
) -> None:
    """Verify a query on Hub A never returns hub-scoped documents belonging only to Hub B."""
    org_id = uuid4()
    hub_a = await create_test_hub(db_session, org_id, "hub-alpha")
    hub_b = await create_test_hub(db_session, org_id, "hub-beta")

    doc_b = await create_test_document(
        db_session, org_id, title="Beta Secrets", scope="hub", hub_id=hub_b.id
    )
    await create_test_chunk(
        db_session, doc_b.id, text="Proprietary deployment strategy for beta team"
    )
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    retriever = HybridRetriever(session_factory, FakeEmbedder(dimension=768))

    # Query targeting Hub A
    ctx_a = AccessContext(org_id=org_id, hub_id=hub_a.id, user_tags=frozenset())
    results_a = await retriever.retrieve("deployment strategy", ctx_a)
    assert len(results_a) == 0

    # Query targeting Hub B
    ctx_b = AccessContext(org_id=org_id, hub_id=hub_b.id, user_tags=frozenset())
    results_b = await retriever.retrieve("deployment strategy", ctx_b)
    assert len(results_b) == 1
    assert results_b[0].document_title == "Beta Secrets"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_global_documents_visible_across_all_hubs(
    db_session: AsyncSession,
) -> None:
    """Verify that documents with scope='global' are visible from any hub in the org."""
    org_id = uuid4()
    hub_1 = await create_test_hub(db_session, org_id, "hub-one")
    hub_2 = await create_test_hub(db_session, org_id, "hub-two")

    doc_global = await create_test_document(
        db_session, org_id, title="Global Employee Handbook", scope="global", hub_id=hub_1.id
    )
    await create_test_chunk(db_session, doc_global.id, text="Company-wide code of conduct rules")
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    retriever = HybridRetriever(session_factory, FakeEmbedder(dimension=768))

    # Query from Hub 2 (which has no direct hub_documents association)
    ctx_2 = AccessContext(org_id=org_id, hub_id=hub_2.id, user_tags=frozenset())
    results = await retriever.retrieve("code of conduct", ctx_2)
    assert len(results) == 1
    assert results[0].document_title == "Global Employee Handbook"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cross_org_isolation_never_returns_other_org_data(
    db_session: AsyncSession,
) -> None:
    """Verify that documents from another organization are never returned."""
    org_1 = uuid4()
    org_2 = uuid4()
    hub_1 = await create_test_hub(db_session, org_1, "org1-hub")
    hub_2 = await create_test_hub(db_session, org_2, "org2-hub")

    # Document in Org 2
    doc_org2 = await create_test_document(
        db_session, org_2, title="Org2 Doc", scope="global", hub_id=hub_2.id
    )
    await create_test_chunk(db_session, doc_org2.id, text="Highly critical database configurations")
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    retriever = HybridRetriever(session_factory, FakeEmbedder(dimension=768))

    # User in Org 1
    ctx_1 = AccessContext(org_id=org_1, hub_id=hub_1.id, user_tags=frozenset())
    results = await retriever.retrieve("database configurations", ctx_1)
    assert len(results) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_inactive_and_superseded_chunks_never_returned(
    db_session: AsyncSession,
) -> None:
    """Verify that inactive chunks and chunks of superseded/failed documents are never returned."""
    org_id = uuid4()
    hub = await create_test_hub(db_session, org_id, "lifecycle-hub")

    # 1. Active document but chunk is_active = False
    doc_active = await create_test_document(db_session, org_id, title="Doc Active", hub_id=hub.id)
    await create_test_chunk(
        db_session, doc_active.id, text="Outdated deactivated paragraph", is_active=False
    )

    # 2. Superseded document
    doc_superseded = await create_test_document(
        db_session, org_id, title="Doc Superseded", status="superseded", hub_id=hub.id
    )
    await create_test_chunk(db_session, doc_superseded.id, text="Old version superseded paragraph")

    # 3. Failed document
    doc_failed = await create_test_document(
        db_session, org_id, title="Doc Failed", status="failed", hub_id=hub.id
    )
    await create_test_chunk(db_session, doc_failed.id, text="Corrupt failed document paragraph")
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    retriever = HybridRetriever(session_factory, FakeEmbedder(dimension=768))

    ctx = AccessContext(org_id=org_id, hub_id=hub.id, user_tags=frozenset())
    results = await retriever.retrieve("paragraph version document", ctx)
    assert len(results) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_metadata_scope_changes_visibility_without_re_embedding(
    db_session: AsyncSession,
) -> None:
    """Verify changing scope between 'hub' and 'global' toggles cross-hub visibility."""
    org_id = uuid4()
    hub_a = await create_test_hub(db_session, org_id, "hub-a")
    hub_b = await create_test_hub(db_session, org_id, "hub-b")

    doc_repo = DocumentRepository(db_session)
    doc = await doc_repo.create(
        org_id=org_id,
        source_uri="file:///policy.md",
        title="Toggle Policy",
        content_hash="hash-123",
        scope="hub",
        owner_hub_id=hub_a.id,
    )
    await doc_repo.add_to_hub(doc.id, hub_a.id)
    await create_test_chunk(db_session, doc.id, text="Standard operating procedure guidelines")
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    retriever = HybridRetriever(session_factory, FakeEmbedder(dimension=768))
    ctx_b = AccessContext(org_id=org_id, hub_id=hub_b.id, user_tags=frozenset())

    # 1. Initial state: scope='hub' -> invisible to Hub B
    results1 = await retriever.retrieve("procedure guidelines", ctx_b)
    assert len(results1) == 0

    # 2. Update metadata to scope='global' -> visible to Hub B without re-embedding
    await doc_repo.update_metadata(doc.id, scope="global")
    await db_session.commit()

    results2 = await retriever.retrieve("procedure guidelines", ctx_b)
    assert len(results2) == 1
    assert results2[0].document_title == "Toggle Policy"

    # 3. Update back to scope='hub' -> invisible to Hub B again
    await doc_repo.update_metadata(doc.id, scope="hub")
    await db_session.commit()

    results3 = await retriever.retrieve("procedure guidelines", ctx_b)
    assert len(results3) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fewer_than_k_chunks_returns_available_without_error(
    db_session: AsyncSession,
) -> None:
    """Verify that fewer than k eligible chunks returns all available results without error."""
    org_id = uuid4()
    hub = await create_test_hub(db_session, org_id, "small-hub")
    doc = await create_test_document(db_session, org_id, title="Small Doc", hub_id=hub.id)
    await create_test_chunk(
        db_session, doc.id, text="Only single matching chunk in small repository"
    )
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    retriever = HybridRetriever(
        session_factory,
        FakeEmbedder(dimension=768),
        config=RetrievalConfig(vector_k=50, fts_k=50, fused_k=30),
    )

    ctx = AccessContext(org_id=org_id, hub_id=hub.id, user_tags=frozenset())
    results = await retriever.retrieve("single matching chunk", ctx)
    assert len(results) == 1
    assert results[0].document_title == "Small Doc"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stop_word_only_query_skips_fts_and_returns_vector_hits(
    db_session: AsyncSession,
) -> None:
    """Verify that stop-word-only or punctuation-only query returns vector hits without crashing."""
    org_id = uuid4()
    hub = await create_test_hub(db_session, org_id, "stopword-hub")
    doc = await create_test_document(db_session, org_id, title="Word Doc", hub_id=hub.id)
    await create_test_chunk(db_session, doc.id, text="General informational content chunk")
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    retriever = HybridRetriever(session_factory, FakeEmbedder(dimension=768))

    ctx = AccessContext(org_id=org_id, hub_id=hub.id, user_tags=frozenset())
    # "the of and" evaluates to empty tsquery in websearch_to_tsquery('english', ...)
    results = await retriever.retrieve("the of and", ctx)
    assert len(results) >= 1
    # FTS score should be None since FTS was skipped, but vector similarity exists
    assert results[0].fts_rank_score is None
    assert results[0].vector_similarity is not None
