"""Integration & fixture-based end-to-end tests for evaluation runner and regression gate."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from typer.testing import CliRunner

from kb.cli import app
from kb.config import Settings
from kb.domain.evaluation import GoldenItem, GoldenType, Provenance, RestrictionKind
from kb.domain.models import DocumentScope
from kb.domain.retrieval import (
    AccessContext,
    ContextConfig,
    RerankConfig,
    RetrievalConfig,
    ScoredChunk,
)
from kb.evaluation.gate import evaluate_gate, save_baseline
from kb.evaluation.golden import write_golden
from kb.evaluation.runner import run_evaluation
from kb.generation.generator import Generator
from kb.ingestion.pipeline import IngestionPipeline, IngestRequest
from kb.llm.client import LLMClient
from kb.llm.fakes import FakeEmbedder, FakeLLM, FakeReranker
from kb.rag.service import RagService
from kb.retrieval.context import ContextAssembler
from kb.retrieval.hybrid import HybridRetriever
from kb.retrieval.rerank import RerankStage
from kb.retrieval.retriever import Retriever
from tests.integration.helpers import create_test_chunk, create_test_document, create_test_hub


def _test_settings(org_id_str: str) -> Settings:
    return Settings(
        DATABASE_URL="postgresql+psycopg://u:p@localhost:5432/db",
        REDIS_URL="redis://localhost:6379/0",
        JWT_SECRET="test-secret-key-minimum-32-chars",
        EMBEDDING_DIM=768,
        DEFAULT_ORG_ID=org_id_str,
        LLM_PROVIDER="fake",
        EMBEDDING_PROVIDER="fake",
        RERANKER_PROVIDER="fake",
        JUDGE_PROVIDER="fake",
    )


@pytest.mark.eval_fixture
@pytest.mark.asyncio
async def test_eval_fixture_end_to_end_and_degraded_gate_failure(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """Fixture E2E: good config passes gate; degraded k=1 drops recall@5 and fails gate."""
    org_id = uuid4()
    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    embedder = FakeEmbedder(dimension=768)
    pipeline = IngestionPipeline(embedder=embedder, session_factory=session_factory)

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    doc1 = corpus_dir / "alpha_policy.html"
    doc1.write_text(
        "<html><body><h1>Alpha Policy</h1>"
        "<p>Quantum falcon protocol requires 99 days retention.</p></body></html>",
        encoding="utf-8",
    )
    doc2 = corpus_dir / "beta_policy.html"
    doc2.write_text(
        "<html><body><h1>Beta Policy</h1>"
        "<p>Emerald phoenix backup runs at 03:00 UTC daily.</p></body></html>",
        encoding="utf-8",
    )
    doc3 = corpus_dir / "secret_finance.html"
    doc3.write_text(
        "<html><body><h1>Finance</h1><p>Secret cobalt ledger surplus is $91M.</p></body></html>",
        encoding="utf-8",
    )

    for fp, tags in [(doc1, []), (doc2, []), (doc3, ["finance"])]:
        await pipeline.ingest_file(
            fp,
            IngestRequest(
                org_id=org_id,
                hub_name="default",
                scope=DocumentScope.HUB,
                acl_tags=tags,
                source_root=corpus_dir,
            ),
        )

    golden_path = tmp_path / "golden.jsonl"
    items = [
        GoldenItem(
            id="q-0001",
            question="quantum falcon retention and emerald phoenix backup?",
            reference_answer="99 days retention and 03:00 UTC daily backup.",
            type=GoldenType.multi_hop,
            source_uris=["alpha_policy.html", "beta_policy.html"],
            evidence_spans=[
                "Quantum falcon protocol requires 99 days retention.",
                "Emerald phoenix backup runs at 03:00 UTC daily.",
            ],
            provenance=Provenance.handwritten,
            reviewed=True,
        ),
        GoldenItem(
            id="q-0002",
            question="What is the secret cobalt ledger surplus?",
            reference_answer="",
            type=GoldenType.access_restricted,
            source_uris=["secret_finance.html"],
            evidence_spans=["Secret cobalt ledger surplus is $91M."],
            restriction_kind=RestrictionKind.acl_tag,
            user_tags=[],
            tags=["acl:finance"],
            provenance=Provenance.manifest,
            reviewed=True,
        ),
        GoldenItem(
            id="q-0003",
            question="Unreviewed draft item?",
            reference_answer="Draft.",
            type=GoldenType.factual,
            source_uris=["alpha_policy.html"],
            evidence_spans=["Quantum falcon protocol requires 99 days retention."],
            provenance=Provenance.handwritten,
            reviewed=False,
        ),
    ]
    write_golden(golden_path, items)

    settings = _test_settings(str(org_id))

    def _make_service(ret_cfg: RetrievalConfig, rerank_cfg: RerankConfig) -> RagService:
        # Scripted FakeLLM extracts first chunk ID from <context> and cites it
        def _fake_ans(prompt: str) -> str:
            import re

            m = re.search(r'<chunk id="([^"]+)"', prompt)
            if "99 days" in prompt and m:
                cid = m.group(1)
                return json.dumps(
                    {
                        "answer": f"99 days retention and 03:00 UTC backup [[chunk:{cid}]].",
                        "citations": [cid],
                        "refusal": False,
                        "refusal_reason": None,
                    }
                )
            return json.dumps(
                {
                    "answer": "Refused.",
                    "citations": [],
                    "refusal": True,
                    "refusal_reason": "insufficient",
                }
            )

        hybrid = HybridRetriever(session_factory, embedder, config=ret_cfg)
        rerank = RerankStage(FakeReranker(), config=rerank_cfg)
        assembler = ContextAssembler(
            config=ContextConfig(budget_tokens=3000), session_factory=session_factory
        )
        gen = Generator(LLMClient(FakeLLM(responses=_fake_ans)), on_uncited="refuse")
        return RagService(Retriever(hybrid, rerank), assembler, gen)

    # 1. Good config run (excludes q-0003 unreviewed item -> item_count == 2)
    good_ret = RetrievalConfig(vector_k=10, fts_k=10, fused_k=10)
    good_rerank = RerankConfig(enabled=True, top_n=5)
    good_report, good_dir = await run_evaluation(
        golden_path=golden_path,
        out_dir=tmp_path / "reports",
        session_factory=session_factory,
        rag_service=_make_service(good_ret, good_rerank),
        settings=settings,
        retrieval_cfg=good_ret,
        rerank_cfg=good_rerank,
        context_cfg=ContextConfig(),
        refusal_min_score=0.0,
        models={"llm": "fake", "embedder": "fake", "reranker": "fake", "judge": "none"},
        judge=None,
        run_preflight=True,
    )
    assert good_report.item_count == 2  # q-0003 excluded!
    assert good_report.summary_metrics["recall@5"].mean == pytest.approx(1.0)
    assert good_report.summary_metrics["access_leaks"].mean == pytest.approx(0.0)

    baseline_path = save_baseline(good_dir, name="fixture", baselines_dir=tmp_path / "baselines")
    thresholds_path = Path("eval/thresholds.json")

    # Baseline run itself passes gate
    base_gate = evaluate_gate(
        report_path=good_dir, baseline_path=baseline_path, thresholds_path=thresholds_path
    )
    assert base_gate.passed is True

    # 2. Degraded config (fused_k=1, top_n=1) retrieves only 1 of 2 spans -> recall@5 = 0.5
    bad_ret = RetrievalConfig(vector_k=10, fts_k=10, fused_k=1)
    bad_rerank = RerankConfig(enabled=True, top_n=1)
    bad_report, bad_dir = await run_evaluation(
        golden_path=golden_path,
        out_dir=tmp_path / "reports",
        session_factory=session_factory,
        rag_service=_make_service(bad_ret, bad_rerank),
        settings=settings,
        retrieval_cfg=bad_ret,
        rerank_cfg=bad_rerank,
        context_cfg=ContextConfig(),
        refusal_min_score=0.0,
        models={"llm": "fake", "embedder": "fake", "reranker": "fake", "judge": "none"},
        judge=None,
        run_preflight=True,
    )
    assert bad_report.summary_metrics["recall@5"].mean == pytest.approx(0.5)

    # Run `kb eval gate` via CLI and assert non-zero exit code
    runner = CliRunner()
    gate_cli = runner.invoke(
        app,
        [
            "eval",
            "gate",
            "--report",
            str(bad_dir),
            "--baseline",
            str(baseline_path),
            "--thresholds",
            str(thresholds_path),
        ],
    )
    assert gate_cli.exit_code != 0
    assert "FAIL" in gate_cli.output


@pytest.mark.integration
@pytest.mark.asyncio
async def test_broken_retriever_triggers_oracle_access_leak_and_fails_gate(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    """A deliberately broken retriever returning an ineligible chunk produces access_leaks >= 1."""
    org_id = uuid4()
    hub = await create_test_hub(db_session, org_id, "default")
    doc = await create_test_document(
        db_session,
        org_id,
        title="Restricted",
        source_uri="res.pdf",
        acl_tags=["finance"],
        hub_id=hub.id,
    )
    leaked_chunk = await create_test_chunk(
        db_session, doc.id, text="Confidential $99M figure.", ordinal=0
    )
    await db_session.commit()

    session_factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)

    class BrokenRetriever(Retriever):
        def __init__(self) -> None:
            pass

        async def retrieve(self, query: str, access: AccessContext) -> list[ScoredChunk]:
            return [
                ScoredChunk(
                    chunk_id=leaked_chunk.id,
                    document_id=doc.id,
                    document_title="Restricted",
                    source_uri="res.pdf",
                    text="Confidential $99M figure.",
                    page=1,
                    section=None,
                    ordinal=0,
                    content_hash="h",
                    vector_similarity=0.9,
                    fts_rank_score=0.9,
                    vector_rank=1,
                    fts_rank=1,
                    fused_score=0.9,
                    rerank_score=0.9,
                    final_score=0.9,
                )
            ]

    golden_path = tmp_path / "golden_leak.jsonl"
    write_golden(
        golden_path,
        [
            GoldenItem(
                id="q-0001",
                question="Confidential figure?",
                reference_answer="",
                type=GoldenType.access_restricted,
                source_uris=["res.pdf"],
                evidence_spans=["Confidential $99M figure."],
                restriction_kind=RestrictionKind.acl_tag,
                user_tags=[],
                tags=["acl:finance"],
                provenance=Provenance.manifest,
                reviewed=True,
            )
        ],
    )
    settings = _test_settings(str(org_id))
    assembler = ContextAssembler(
        config=ContextConfig(budget_tokens=3000), session_factory=session_factory
    )
    gen = Generator(
        LLMClient(
            FakeLLM(
                responses=[
                    json.dumps(
                        {
                            "answer": "Refused.",
                            "citations": [],
                            "refusal": True,
                            "refusal_reason": "no",
                        }
                    )
                ]
            )
        )
    )
    svc = RagService(BrokenRetriever(), assembler, gen)

    report, report_dir = await run_evaluation(
        golden_path=golden_path,
        out_dir=tmp_path / "reports",
        session_factory=session_factory,
        rag_service=svc,
        settings=settings,
        retrieval_cfg=RetrievalConfig(),
        rerank_cfg=RerankConfig(),
        context_cfg=ContextConfig(),
        refusal_min_score=0.0,
        models={"llm": "fake", "embedder": "fake", "reranker": "fake", "judge": "none"},
        include_unreviewed=False,
        run_preflight=False,
    )
    assert report.summary_metrics["access_leaks"].mean == pytest.approx(1.0)
    gate_res = evaluate_gate(
        report_path=report_dir,
        baseline_path=tmp_path / "nonexistent.json",
        thresholds_path=Path("eval/thresholds.json"),
    )
    assert gate_res.passed is False
    assert any("access_leaks" in hf for hf in gate_res.hard_failures)


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_eval_run_with_real_providers(tmp_path: Path) -> None:
    """Live E2E eval run over 2 items when real Ollama + Judge daemons are reachable."""
    import urllib.request

    from kb.config import get_settings
    from kb.db.session import get_session_factory
    from kb.evaluation.metrics.judge import NativeJudgeEvaluator
    from kb.evaluation.runner import apply_config_overrides
    from kb.llm.factory import build_embedder, build_judge, build_llm, build_reranker

    settings = get_settings()
    if settings.LLM_PROVIDER != "ollama" or not settings.JUDGE_PROVIDER:
        pytest.skip("Real LLM/Judge providers not configured in settings")

    try:
        urllib.request.urlopen(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
    except Exception:
        pytest.skip("Ollama daemon not reachable")

    ret_cfg, rerank_cfg, ctx_cfg, min_score = apply_config_overrides(settings, {})
    sf = get_session_factory()
    hybrid = HybridRetriever(sf, build_embedder(settings), config=ret_cfg)
    retriever = Retriever(hybrid, RerankStage(build_reranker(settings), config=rerank_cfg))
    assembler = ContextAssembler(config=ctx_cfg, session_factory=sf)
    llm = build_llm(settings)
    judge_llm = build_judge(settings)
    svc = RagService(retriever, assembler, Generator(LLMClient(llm)), refusal_min_score=min_score)

    report, _ = await run_evaluation(
        golden_path=Path("eval/golden/golden.jsonl"),
        out_dir=tmp_path / "live_reports",
        session_factory=sf,
        rag_service=svc,
        settings=settings,
        retrieval_cfg=ret_cfg,
        rerank_cfg=rerank_cfg,
        context_cfg=ctx_cfg,
        refusal_min_score=min_score,
        models={
            "llm": settings.LLM_MODEL,
            "embedder": settings.EMBEDDING_MODEL,
            "reranker": settings.RERANKER_PROVIDER,
            "judge": settings.JUDGE_MODEL,
        },
        judge=NativeJudgeEvaluator(LLMClient(judge_llm)),
        include_unreviewed=True,
        limit=2,
        concurrency=1,
        run_preflight=False,
    )
    assert report.item_count == 2
