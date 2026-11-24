"""End-to-end integration tests for RAG cited generation and access isolation."""

import json
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kb.domain.answers import RefusalReason
from kb.domain.retrieval import AccessContext, ContextConfig, RerankConfig
from kb.generation.generator import Generator
from kb.llm.client import LLMClient
from kb.llm.fakes import FakeEmbedder, FakeLLM, FakeReranker
from kb.rag.service import RagService
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
async def test_access_isolated_rag_query(
    db_session: AsyncSession,
) -> None:
    """Verify that a user without the 'finance' tag never has finance chunks in prompt or answer."""
    org_id = uuid4()
    hub = await create_test_hub(db_session, org_id, "finance-hub")

    vec = make_deterministic_embedding("q3 revenue and net margins")

    # Document protected by 'finance' tag
    doc_finance = await create_test_document(
        db_session,
        org_id,
        title="Q3 Confidential Financials",
        acl_tags=["finance"],
        hub_id=hub.id,
    )
    chunk_finance = await create_test_chunk(
        db_session,
        doc_finance.id,
        text="Confidential: Q3 net margin was 42 percent with revenue of 120 million dollars.",
        embedding=vec,
    )
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    fake_embedder = FakeEmbedder(dimension=768)
    fake_embedder.embed_map = {"q3 revenue": vec}

    hybrid_retriever = HybridRetriever(session_factory, fake_embedder)
    rerank_stage = RerankStage(FakeReranker(), RerankConfig(enabled=True, top_n=5))
    retriever = Retriever(hybrid_retriever, rerank_stage)
    assembler = ContextAssembler(
        config=ContextConfig(budget_tokens=2000),
        session_factory=session_factory,
    )

    # 1. Query by user WITHOUT finance tag
    unauthorized_ctx = AccessContext(
        org_id=org_id, hub_id=hub.id, user_tags=frozenset(["engineering"])
    )

    fake_llm = FakeLLM()
    client = LLMClient(fake_llm)
    generator = Generator(client)
    rag_service = RagService(retriever, assembler, generator)

    unauthorized_result = await rag_service.ask("What was the Q3 revenue?", unauthorized_ctx)

    # Should refuse before calling LLM due to empty context
    assert unauthorized_result.refusal is True
    assert unauthorized_result.refusal_reason == RefusalReason.no_context
    assert len(fake_llm.history) == 0
    assert "42 percent" not in unauthorized_result.answer

    # 2. Query by user WITH finance tag
    authorized_ctx = AccessContext(org_id=org_id, hub_id=hub.id, user_tags=frozenset(["finance"]))

    answer_json = json.dumps(
        {
            "answer": f"Q3 net margin was 42% with $120M revenue [[chunk:{chunk_finance.id}]].",
            "citations": [str(chunk_finance.id)],
            "refusal": False,
            "refusal_reason": None,
        }
    )
    fake_llm.responses = [answer_json]

    authorized_result = await rag_service.ask("What was the Q3 revenue?", authorized_ctx)

    assert authorized_result.refusal is False
    assert len(authorized_result.citations) == 1
    assert authorized_result.citations[0].chunk_id == chunk_finance.id
    assert "42 percent" in authorized_result.citations[0].snippet
    assert len(fake_llm.history) == 1
    # Chunk text was delivered in context
    assert "42 percent" in fake_llm.history[0]["prompt"]


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_ollama_answer_generation() -> None:
    """Live test running against a local Ollama daemon (skipped if offline)."""
    import urllib.error
    import urllib.request

    from kb.llm.ollama import OllamaLLM

    # Probe Ollama daemon
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            _ = [m.get("name") for m in data.get("models", [])]
    except Exception:
        pytest.skip("Local Ollama daemon not reachable at http://localhost:11434")

    ollama_llm = OllamaLLM(base_url="http://localhost:11434", model="llama3.2:3b")
    response = await ollama_llm.generate("Say hello in one word.")
    assert len(response.content) > 0
