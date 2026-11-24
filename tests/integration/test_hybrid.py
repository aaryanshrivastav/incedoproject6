"""Integration tests for hybrid retrieval, RRF fusion, and iterative HNSW scan."""

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kb.domain.models import NewChunk
from kb.domain.retrieval import AccessContext, RetrievalConfig, RetrievalError
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
async def test_iterative_scan_assert_active_and_returns_k_eligible_under_filter(
    db_session: AsyncSession,
) -> None:
    """Verify that HNSW iterative scan returns all eligible chunks when most are filtered out."""
    org_id = uuid4()
    hub_eligible = await create_test_hub(db_session, org_id, "hub-eligible")
    hub_ineligible = await create_test_hub(db_session, org_id, "hub-ineligible")

    # Target query vector
    query_text = "strategic architecture guidelines"
    query_vec = make_deterministic_embedding(query_text)

    # 1. Create 5 ELIGIBLE chunks in hub_eligible
    doc_eligible = await create_test_document(
        db_session, org_id, title="Eligible Architecture", hub_id=hub_eligible.id
    )
    for i in range(5):
        # Slightly perturbed vectors still matching query
        chunk_vec = list(query_vec)
        chunk_vec[i] += 0.001
        norm = sum(v * v for v in chunk_vec) ** 0.5
        chunk_vec = [v / norm for v in chunk_vec]

        await create_test_chunk(
            db_session,
            document_id=doc_eligible.id,
            text=f"Eligible architecture content block number {i}",
            ordinal=i + 1,
            embedding=chunk_vec,
        )

    # 2. Create 150 INELIGIBLE chunks in hub_ineligible that have IDENTICAL vectors (distance 0.0).
    # Under standard ANN without iterative scan, search would fill all k slots with distractors!
    doc_ineligible = await create_test_document(
        db_session, org_id, title="Ineligible Other Hub", hub_id=hub_ineligible.id
    )
    ineligible_chunks: list[NewChunk] = []
    for j in range(150):
        ineligible_chunks.append(
            NewChunk(
                ordinal=j + 1,
                text=f"Ineligible distractor text chunk {j}",
                page=1,
                section="Distractor",
                token_count=5,
                content_hash=f"ineligible-hash-{j}",
                metadata={"distractor": True},
                embedding=query_vec,  # Exact distance 0.0 to query
            )
        )
    from kb.db.repositories import ChunkRepository

    chunk_repo = ChunkRepository(db_session)
    await chunk_repo.add_many(doc_ineligible.id, ineligible_chunks)
    await db_session.commit()

    # 3. Verify iterative scan setting inside transaction
    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    async with session_factory() as s:
        await s.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order;"))
        show_res = await s.execute(text("SHOW hnsw.iterative_scan;"))
        setting_val = show_res.scalar_one()
        assert setting_val == "relaxed_order"

    # 4. Execute search with vector_k=5 on hub_eligible
    fake_embedder = FakeEmbedder(dimension=768)
    fake_embedder.embed_map = {query_text: query_vec}
    retriever = HybridRetriever(
        session_factory,
        fake_embedder,
        config=RetrievalConfig(vector_k=5, fts_k=5, fused_k=5),
    )

    ctx = AccessContext(org_id=org_id, hub_id=hub_eligible.id, user_tags=frozenset())
    results = await retriever.retrieve(query_text, ctx)

    # Must return all 5 eligible chunks despite 150 closer distractor chunks
    assert len(results) == 5
    for r in results:
        assert r.document_title == "Eligible Architecture"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hybrid_retrieval_determinism(
    db_session: AsyncSession,
) -> None:
    """Verify that identical calls to retrieve return identical chunk order and scores."""
    org_id = uuid4()
    hub = await create_test_hub(db_session, org_id, "det-hub")
    doc = await create_test_document(
        db_session, org_id, title="Deterministic Document", hub_id=hub.id
    )
    await create_test_chunk(
        db_session, doc.id, text="Alpha unique content for determinism testing", ordinal=1
    )
    await create_test_chunk(
        db_session, doc.id, text="Beta unique content for determinism testing", ordinal=2
    )
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    retriever = HybridRetriever(session_factory, FakeEmbedder(dimension=768))

    ctx = AccessContext(org_id=org_id, hub_id=hub.id, user_tags=frozenset())
    run1 = await retriever.retrieve("determinism testing", ctx)
    run2 = await retriever.retrieve("determinism testing", ctx)

    assert len(run1) == len(run2)
    assert [r.chunk_id for r in run1] == [r.chunk_id for r in run2]
    assert [r.fused_score for r in run1] == [r.fused_score for r in run2]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wrong_dimension_embedder_raises_retrieval_error(
    db_session: AsyncSession,
) -> None:
    """Verify that an embedder returning the wrong dimension raises a typed RetrievalError."""
    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    # Expected is 768, provide 512
    wrong_embedder = FakeEmbedder(dimension=512)
    retriever = HybridRetriever(session_factory, wrong_embedder)

    ctx = AccessContext(org_id=uuid4(), hub_id=uuid4(), user_tags=frozenset())
    with pytest.raises(RetrievalError, match="Query vector dimension 512 does not match expected"):
        await retriever.retrieve("any query", ctx)
