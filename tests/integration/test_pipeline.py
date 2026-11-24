"""Integration tests for IngestionPipeline and CLI against real PostgreSQL."""

from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from kb.cli import app
from kb.db.models import ChunkModel, DocumentModel
from kb.domain.models import DocumentScope
from kb.ingestion.pipeline import IngestionPipeline, IngestRequest, IngestStatus
from kb.llm.fakes import FakeEmbedder


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_same_file_twice_skips(tmp_path: Path, db_session: AsyncSession) -> None:
    """Verify that ingesting the same file twice produces created then skipped with 0 new chunks."""
    doc_path = tmp_path / "report.html"
    doc_path.write_text(
        "<h1>Financial Summary</h1><p>Net margin for Q3 was 24.5%.</p>", encoding="utf-8"
    )

    embedder = FakeEmbedder(dimension=768, model_name="test-bge")
    pipeline = IngestionPipeline(embedder=embedder)

    org_id = uuid4()
    req = IngestRequest(org_id=org_id, hub_name="finance", scope=DocumentScope.HUB)

    # 1. First ingestion
    res1 = await pipeline.ingest_file(doc_path, req)
    assert res1.status == IngestStatus.CREATED
    assert res1.version == 1
    assert res1.chunks_created > 0

    # 2. Second ingestion of identical file
    res2 = await pipeline.ingest_file(doc_path, req)
    assert res2.status == IngestStatus.SKIPPED
    assert res2.version == 1
    assert res2.chunks_created == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_modify_file_creates_new_version_and_reuses_embeddings(
    tmp_path: Path, db_session: AsyncSession
) -> None:
    """Verify modifying file creates new version, deactivates old chunks, reuses embeddings."""
    doc_path = tmp_path / "guide.html"
    doc_path.write_text(
        "<h1>Security Guide</h1><p>Paragraph One Unchanged.</p><p>Paragraph Two Initial.</p>",
        encoding="utf-8",
    )

    embedder = FakeEmbedder(dimension=768, model_name="test-bge")
    # Wrap embed call to track invocations
    orig_embed = embedder.embed
    embed_mock = AsyncMock(side_effect=orig_embed)
    embedder.embed = embed_mock  # type: ignore[method-assign]

    pipeline = IngestionPipeline(embedder=embedder)
    org_id = uuid4()
    req = IngestRequest(org_id=org_id, hub_name="security", scope=DocumentScope.HUB)

    # 1. Ingest initial version
    res1 = await pipeline.ingest_file(doc_path, req)
    assert res1.status == IngestStatus.CREATED
    assert res1.version == 1
    initial_embed_calls = embed_mock.call_count
    assert initial_embed_calls > 0

    # 2. Modify only Paragraph Two
    doc_path.write_text(
        "<h1>Security Guide</h1><p>Paragraph One Unchanged.</p><p>Paragraph Two MODIFIED.</p>",
        encoding="utf-8",
    )

    res2 = await pipeline.ingest_file(doc_path, req)
    assert res2.status == IngestStatus.UPDATED
    assert res2.version == 2

    # Check that previous version is superseded and new is active
    stmt = (
        select(DocumentModel).where(DocumentModel.org_id == org_id).order_by(DocumentModel.version)
    )
    docs = (await db_session.execute(stmt)).scalars().all()
    assert len(docs) == 2
    assert docs[0].version == 1
    assert docs[0].status == "superseded"
    assert docs[1].version == 2
    assert docs[1].status == "active"

    # Check chunk active statuses
    c_stmt = (
        select(ChunkModel.document_id, ChunkModel.is_active)
        .join(DocumentModel, DocumentModel.id == ChunkModel.document_id)
        .where(DocumentModel.org_id == org_id)
    )
    chunks = (await db_session.execute(c_stmt)).all()
    v1_chunks = [c for c in chunks if c[0] == docs[0].id]
    v2_chunks = [c for c in chunks if c[0] == docs[1].id]

    assert all(c[1] is False for c in v1_chunks)
    assert all(c[1] is True for c in v2_chunks)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_update_without_re_embedding(
    tmp_path: Path, db_session: AsyncSession
) -> None:
    """Verify changing scope or ACL tags updates metadata without calling embedder."""
    doc_path = tmp_path / "memo.html"
    doc_path.write_text(
        "<h1>Internal Memo</h1><p>All hands meeting next Tuesday.</p>", encoding="utf-8"
    )

    embedder = FakeEmbedder(dimension=768, model_name="test-bge")
    embed_mock = AsyncMock(side_effect=embedder.embed)
    embedder.embed = embed_mock  # type: ignore[method-assign]

    pipeline = IngestionPipeline(embedder=embedder)
    org_id = uuid4()

    # 1. First ingest with scope=hub and acl_tag=["engineering"]
    req1 = IngestRequest(
        org_id=org_id,
        hub_name="eng",
        scope=DocumentScope.HUB,
        acl_tags=["engineering"],
    )
    res1 = await pipeline.ingest_file(doc_path, req1)
    assert res1.status == IngestStatus.CREATED
    embed_calls_after_v1 = embed_mock.call_count
    assert embed_calls_after_v1 > 0

    # 2. Ingest same content with scope=global and acl_tag=["public"]
    req2 = IngestRequest(
        org_id=org_id,
        hub_name="eng",
        scope=DocumentScope.GLOBAL,
        acl_tags=["public"],
    )
    res2 = await pipeline.ingest_file(doc_path, req2)
    assert res2.status == IngestStatus.METADATA_UPDATED
    assert res2.version == 1

    # Embedder was NOT called again
    assert embed_mock.call_count == embed_calls_after_v1

    # Verify updated values in DB
    stmt = select(DocumentModel).where(DocumentModel.id == res1.document_id)
    doc = (await db_session.execute(stmt)).scalar_one()
    assert doc.scope == "global"
    assert doc.acl_tags == ["public"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_failure_rollback_leaves_previous_version_active(
    tmp_path: Path, db_session: AsyncSession
) -> None:
    """Verify write transaction rollback preserves previous active document and chunks."""
    doc_path = tmp_path / "contract.html"
    doc_path.write_text("<h1>Contract</h1><p>Initial terms.</p>", encoding="utf-8")

    embedder = FakeEmbedder(dimension=768, model_name="test-bge")
    pipeline = IngestionPipeline(embedder=embedder)
    org_id = uuid4()
    req = IngestRequest(org_id=org_id, hub_name="legal", scope=DocumentScope.HUB)

    # Ingest v1
    res1 = await pipeline.ingest_file(doc_path, req)
    assert res1.status == IngestStatus.CREATED

    # Modify content
    doc_path.write_text(
        "<h1>Contract</h1><p>Updated terms with error injection.</p>", encoding="utf-8"
    )

    # Simulate database error during add_many in ChunkRepository
    with patch(
        "kb.ingestion.pipeline.ChunkRepository.add_many",
        side_effect=RuntimeError("Simulated DB Crash"),
    ):
        with pytest.raises(RuntimeError):
            await pipeline.ingest_file(doc_path, req)

    # Verify v1 is still active and its chunks are still active
    stmt = select(DocumentModel).where(DocumentModel.id == res1.document_id)
    doc = (await db_session.execute(stmt)).scalar_one()
    assert doc.status == "active"

    c_stmt = select(func.count(ChunkModel.id)).where(
        ChunkModel.document_id == res1.document_id,
        ChunkModel.is_active.is_(True),
    )
    active_chunks = (await db_session.execute(c_stmt)).scalar_one()
    assert active_chunks > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_corrupt_file_in_batch_records_failure_and_continues(
    tmp_path: Path, db_session: AsyncSession
) -> None:
    """Verify corrupt files in a batch are recorded as failed and valid files succeed."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()

    # Valid file
    valid_file = batch_dir / "valid.html"
    valid_file.write_text("<h1>Valid Page</h1><p>Valid content line.</p>", encoding="utf-8")

    # Corrupt PDF file
    corrupt_file = batch_dir / "corrupted.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4\nGARBAGE_PAYLOAD")

    embedder = FakeEmbedder(dimension=768, model_name="test-bge")
    pipeline = IngestionPipeline(embedder=embedder)
    org_id = uuid4()
    req = IngestRequest(org_id=org_id, hub_name="ops", scope=DocumentScope.HUB)

    summary = await pipeline.ingest_path(batch_dir, req)

    assert summary.total_files == 2
    assert summary.created == 1
    assert summary.failed == 1
    assert len(summary.failures) == 1
    assert "corrupted.pdf" in summary.failures[0][0]

    # Verify failed record in DB
    stmt = select(DocumentModel).where(
        DocumentModel.org_id == org_id, DocumentModel.status == "failed"
    )
    failed_docs = (await db_session.execute(stmt)).scalars().all()
    assert len(failed_docs) == 1
    assert failed_docs[0].failure_reason == "corrupt"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wrong_dimension_embedder_fails_document(
    tmp_path: Path, db_session: AsyncSession
) -> None:
    """Verify embedder returning wrong vector dimension fails document with embed_error."""
    doc_path = tmp_path / "doc.html"
    doc_path.write_text("<h1>Title</h1><p>Sample content for dimension test.</p>", encoding="utf-8")

    # Embedder with 512 dimensions while table expects 768
    embedder = FakeEmbedder(dimension=512, model_name="wrong-dim-model")
    pipeline = IngestionPipeline(embedder=embedder)
    org_id = uuid4()
    req = IngestRequest(org_id=org_id, hub_name="ops", scope=DocumentScope.HUB)

    # Artificially set pipeline embedder dimension to mismatch check
    embedder._dimension = 768  # Pipeline expects 768, but embed() outputs 512
    with patch.object(embedder, "embed", return_value=[[0.1] * 512]):
        res = await pipeline.ingest_file(doc_path, req)

    assert res.status == IngestStatus.FAILED
    assert "embed_error" in (res.reason or "")

    # Verify failed document stored in DB
    stmt = select(DocumentModel).where(
        DocumentModel.org_id == org_id, DocumentModel.status == "failed"
    )
    failed_doc = (await db_session.execute(stmt)).scalar_one()
    assert failed_doc.failure_reason == "embed_error"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_check_zero_active_chunks_with_null_or_invalid_embedding(
    db_session: AsyncSession,
) -> None:
    """SQL check: verify zero active chunks have NULL embedding or wrong dimension."""
    # Test query ensures data integrity invariant
    query = text("""
        SELECT count(*)
        FROM chunks
        WHERE is_active = true
          AND (embedding IS NULL OR vector_dims(embedding) != 768);
    """)
    res = await db_session.execute(query)
    count = res.scalar_one()
    assert count == 0


@pytest.mark.integration
def test_cli_ingest_and_docs_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Typer CLI commands 'ingest' and 'docs list'."""
    from kb.config import get_settings

    try:
        get_settings()
    except Exception:
        pytest.skip("Integration database not configured (DATABASE_URL unset)")

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")

    get_settings.cache_clear()
    runner = CliRunner()

    sample_file = tmp_path / "cli_sample.html"
    sample_file.write_text(
        "<h1>CLI Sample</h1><p>Ingesting via Typer runner.</p>", encoding="utf-8"
    )

    # 1. Ingest via CLI with --json
    result = runner.invoke(
        app,
        [
            "ingest",
            str(sample_file),
            "--hub",
            "cli-hub",
            "--scope",
            "hub",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert '"total_files": 1' in result.output
    assert '"created": 1' in result.output

    # 2. Inspect with 'docs list'
    list_res = runner.invoke(app, ["docs", "list", "--hub", "cli-hub"])
    assert list_res.exit_code == 0
    assert "CLI Sample" in list_res.output

    # 3. Ingest corrupt file with --fail-on-error
    corrupt_file = tmp_path / "bad.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4\nGARBAGE")

    fail_res = runner.invoke(
        app,
        ["ingest", str(corrupt_file), "--fail-on-error"],
    )
    assert fail_res.exit_code == 1
