"""Integration tests for corpus setup idempotency, check-corpus preflight, and LLM sampling."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kb.domain.evaluation import GoldenItem, GoldenType, Provenance, RestrictionKind
from kb.domain.models import DocumentScope
from kb.evaluation.corpus_plan import (
    CorpusPlan,
    CorpusPlanGroup,
    check_corpus_against_golden,
)
from kb.evaluation.generate import from_chunks_llm
from kb.ingestion.pipeline import IngestionPipeline, IngestRequest
from kb.llm.client import LLMClient
from kb.llm.fakes import FakeEmbedder, FakeLLM
from tests.integration.helpers import create_test_chunk, create_test_document, create_test_hub


@pytest.mark.integration
@pytest.mark.asyncio
async def test_corpus_setup_idempotency_and_check_corpus_errors(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """Ingestion via corpus plan is idempotent and check_corpus detects distinct failure modes."""
    org_id = uuid4()
    doc_file = tmp_path / "policy.html"
    doc_file.write_text(
        "<html><body><h1>Policy</h1><p>Alpha beta gamma delta.</p></body></html>",
        encoding="utf-8",
    )
    plan = CorpusPlan(
        groups=[
            CorpusPlanGroup(
                glob="policy.html",
                hub="default",
                scope=DocumentScope.HUB,
                acl_tags=["finance"],
            )
        ]
    )
    assert len(plan.groups) == 1

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    pipeline = IngestionPipeline(
        embedder=FakeEmbedder(dimension=768), session_factory=session_factory
    )
    req = IngestRequest(
        org_id=org_id,
        hub_name="default",
        scope=DocumentScope.HUB,
        acl_tags=["finance"],
        source_root=tmp_path,
    )

    first = await pipeline.ingest_file(doc_file, req)
    second = await pipeline.ingest_file(doc_file, req)
    assert first.chunks_created >= 1
    assert second.chunks_created == 0  # idempotent (I5)

    # Create a multi-chunk document to test chunk-boundary span error
    hub = await create_test_hub(db_session, org_id, "default")
    split_doc = await create_test_document(
        db_session,
        org_id,
        title="Split Doc",
        source_uri="split.pdf",
        acl_tags=[],
        hub_id=hub.id,
    )
    await create_test_chunk(db_session, split_doc.id, text="First chunk ends here", ordinal=0)
    await create_test_chunk(db_session, split_doc.id, text="Second chunk starts here", ordinal=1)
    await db_session.commit()

    items = [
        # 1. Missing document
        GoldenItem(
            id="q-0001",
            question="Missing doc?",
            reference_answer="Ans",
            type=GoldenType.factual,
            source_uris=["nonexistent.pdf"],
            evidence_spans=["missing"],
            provenance=Provenance.handwritten,
        ),
        # 2. Span not found
        GoldenItem(
            id="q-0002",
            question="Missing span?",
            reference_answer="Ans",
            type=GoldenType.factual,
            source_uris=["split.pdf"],
            evidence_spans=["completely absent phrase"],
            provenance=Provenance.handwritten,
        ),
        # 3. Span crosses chunk boundary
        GoldenItem(
            id="q-0003",
            question="Boundary span?",
            reference_answer="Ans",
            type=GoldenType.factual,
            source_uris=["split.pdf"],
            evidence_spans=["ends here Second chunk"],
            provenance=Provenance.handwritten,
        ),
        # 4. Wrong ACL metadata (policy.html has ['finance'], plan expects ['hr'])
        GoldenItem(
            id="q-0004",
            question="Wrong ACL?",
            reference_answer="",
            type=GoldenType.access_restricted,
            source_uris=["policy.html"],
            evidence_spans=["Alpha beta gamma delta."],
            restriction_kind=RestrictionKind.acl_tag,
            user_tags=["finance"],
            provenance=Provenance.manifest,
        ),
    ]

    bad_plan = CorpusPlan(
        groups=[
            CorpusPlanGroup(
                glob="policy.html",
                hub="other",
                scope=DocumentScope.HUB,
                acl_tags=["hr"],
            )
        ]
    )
    errors = await check_corpus_against_golden(
        db_session, org_id, items, plan=bad_plan, root=tmp_path
    )
    joined = "\n".join(errors)
    assert "Missing active document" in joined
    assert "span not found" in joined
    assert "span crosses chunk boundary" in joined
    assert "has acl_tags" in joined
    assert "missing expected hub membership" in joined


@pytest.mark.integration
@pytest.mark.asyncio
async def test_from_chunks_llm_deterministic_and_substring_filter(
    db_session: AsyncSession,
) -> None:
    """from_chunks_llm keeps exact substring spans, discards hallucinated spans, and is seeded."""
    org_id = uuid4()
    hub = await create_test_hub(db_session, org_id, "default")
    doc = await create_test_document(
        db_session, org_id, title="LLM Doc", source_uri="llm_doc.pdf", hub_id=hub.id
    )
    await create_test_chunk(
        db_session,
        doc.id,
        text="Northwind Labs mandates 30-day key rotation for all service tokens.",
        ordinal=0,
    )
    await create_test_chunk(
        db_session,
        doc.id,
        text="All Kubernetes pods must define explicit memory limits.",
        ordinal=1,
    )
    await db_session.commit()

    valid_json = json.dumps(
        {
            "question": "What is the key rotation interval?",
            "reference_answer": "Service tokens require 30-day key rotation.",
            "evidence_span": "30-day key rotation for all service tokens",
        }
    )
    invalid_span_json = json.dumps(
        {
            "question": "Hallucinated span?",
            "reference_answer": "Yes.",
            "evidence_span": "this substring does not exist in chunk",
        }
    )

    # Use a callable FakeLLM so whichever of the 2 chunks is sampled first gets the valid span
    # matching that chunk, and the second gets the hallucinated non-substring span.
    def _scripted_response(prompt: str) -> str:
        if "30-day key rotation" in prompt:
            return valid_json
        return invalid_span_json

    llm1 = FakeLLM(responses=_scripted_response)
    res1 = await from_chunks_llm(db_session, LLMClient(llm1), n=2, seed=7, org_id=org_id)

    llm2 = FakeLLM(responses=_scripted_response)
    res2 = await from_chunks_llm(db_session, LLMClient(llm2), n=2, seed=7, org_id=org_id)

    # First chunk had valid substring, second had invalid -> only 1 kept
    assert len(res1) == 1
    assert res1[0].provenance == Provenance.llm
    assert res1[0].reviewed is False
    assert [r.question for r in res1] == [r.question for r in res2]
