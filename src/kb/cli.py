"""Command line interface for the knowledge assistant."""

import asyncio
import json
from pathlib import Path
from uuid import UUID

import typer

from kb import __version__
from kb.config import get_settings
from kb.db.repositories import DocumentRepository, HubRepository
from kb.db.session import session_scope
from kb.domain.models import DocumentScope
from kb.domain.retrieval import AccessContext, ContextConfig, RerankConfig
from kb.generation.generator import Generator
from kb.ingestion.pipeline import IngestionPipeline, IngestRequest
from kb.llm.client import LLMClient
from kb.llm.factory import build_embedder, build_llm, build_reranker
from kb.rag.service import RagService
from kb.retrieval.context import ContextAssembler
from kb.retrieval.hybrid import HybridRetriever
from kb.retrieval.rerank import RerankStage
from kb.retrieval.retriever import Retriever

app = typer.Typer(
    name="kb",
    help="Enterprise GenAI Knowledge Assistant CLI",
    no_args_is_help=True,
)

docs_app = typer.Typer(
    name="docs",
    help="Inspect ingested knowledge repository documents",
    no_args_is_help=True,
)
app.add_typer(docs_app, name="docs")


@app.callback()
def main() -> None:
    """Enterprise GenAI Knowledge Assistant."""


@app.command()
def version() -> None:
    """Print the application version."""
    typer.echo(f"kb version {__version__}")


@app.command(name="ingest")
def ingest(
    path: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to file or directory to ingest",
    ),
    hub: str = typer.Option(
        "default",
        "--hub",
        "-h",
        help="Target hub name",
    ),
    scope: str = typer.Option(
        "hub",
        "--scope",
        "-s",
        help="Document scope: 'hub' or 'global'",
    ),
    acl_tag: list[str] = typer.Option(
        None,
        "--acl-tag",
        "-t",
        help="Access control tag (repeatable)",
    ),
    org_id: UUID | None = typer.Option(
        None,
        "--org-id",
        help="Organization UUID (defaults to DEFAULT_ORG_ID)",
    ),
    source_root: Path | None = typer.Option(
        None,
        "--source-root",
        help="Base root directory for relative source_uri computation",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Output raw JSON summary",
    ),
    fail_on_error: bool = typer.Option(
        False,
        "--fail-on-error",
        help="Exit with non-zero status if any files fail",
    ),
) -> None:
    """Ingest a file or directory of documents into the knowledge repository."""
    settings = get_settings()
    effective_org_id = org_id or settings.DEFAULT_ORG_ID

    try:
        doc_scope = DocumentScope(scope.lower())
    except ValueError:
        typer.secho(
            f"Error: Invalid scope '{scope}'. Must be 'hub' or 'global'.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2) from None

    embedder = build_embedder(settings)
    pipeline = IngestionPipeline(embedder=embedder)

    request = IngestRequest(
        org_id=effective_org_id,
        hub_name=hub,
        scope=doc_scope,
        acl_tags=acl_tag or [],
        source_root=source_root,
    )

    summary = asyncio.run(pipeline.ingest_path(path, request))

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "total_files": summary.total_files,
                    "created": summary.created,
                    "updated": summary.updated,
                    "skipped": summary.skipped,
                    "metadata_updated": summary.metadata_updated,
                    "failed": summary.failed,
                    "chunks_created": summary.chunks_created,
                    "failures": summary.failures,
                },
                indent=2,
            )
        )
    else:
        typer.echo("\n--- Ingestion Summary ---")
        typer.echo(f"  Total Files Processed : {summary.total_files}")
        typer.echo(f"  Documents Created     : {summary.created}")
        typer.echo(f"  Documents Updated     : {summary.updated}")
        typer.echo(f"  Documents Skipped     : {summary.skipped}")
        typer.echo(f"  Metadata Updated      : {summary.metadata_updated}")
        typer.echo(f"  Documents Failed      : {summary.failed}")
        typer.echo(f"  Total Chunks Created  : {summary.chunks_created}")

        if summary.failures:
            typer.secho("\nFailures:", fg=typer.colors.RED, bold=True)
            for uri, reason in summary.failures:
                typer.echo(f"  - {uri}: {reason}")

    if fail_on_error and summary.failed > 0:
        raise typer.Exit(code=1)


@docs_app.command(name="list")
def list_docs(
    hub: str | None = typer.Option(
        None,
        "--hub",
        "-h",
        help="Filter by hub name",
    ),
    status: str | None = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by document status (active, superseded, failed)",
    ),
    org_id: UUID | None = typer.Option(
        None,
        "--org-id",
        help="Organization UUID (defaults to DEFAULT_ORG_ID)",
    ),
) -> None:
    """List documents stored in the knowledge repository."""
    settings = get_settings()
    effective_org_id = org_id or settings.DEFAULT_ORG_ID

    async def _list() -> None:
        async with session_scope() as session:
            doc_repo = DocumentRepository(session)
            hub_repo = HubRepository(session)

            hub_id = None
            if hub is not None:
                hub_model = await hub_repo.get_or_create(effective_org_id, hub)
                hub_id = hub_model.id

            docs = await doc_repo.list(effective_org_id, hub_id=hub_id, status=status)

            if not docs:
                typer.echo("No documents found matching criteria.")
                return

            header = (
                f"{'ID':<38} {'VER':<5} {'STATUS':<12} {'SCOPE':<8} {'TITLE':<25} {'SOURCE URI'}"
            )
            typer.echo(header)
            typer.echo("-" * len(header))
            for d in docs:
                title_preview = d.title[:23] + ".." if len(d.title) > 25 else d.title
                doc_id_str = str(d.id)
                typer.echo(
                    f"{doc_id_str:<38} {d.version:<5} {d.status:<12} {d.scope:<8} "
                    f"{title_preview:<25} {d.source_uri}"
                )

    asyncio.run(_list())


@app.command(name="ask")
def ask(
    question: str = typer.Argument(
        ...,
        help="Question to ask the knowledge assistant",
    ),
    hub: str = typer.Option(
        "default",
        "--hub",
        "-h",
        help="Target hub name",
    ),
    user_tags: list[str] = typer.Option(
        None,
        "--user-tag",
        "-t",
        help="User access control tag (repeatable)",
    ),
    org_id: UUID | None = typer.Option(
        None,
        "--org-id",
        "-o",
        help="Organization UUID (defaults to DEFAULT_ORG_ID from settings)",
    ),
    no_rerank: bool = typer.Option(
        False,
        "--no-rerank",
        help="Disable reranker stage and rely solely on fused hybrid retrieval",
    ),
    show_context: bool = typer.Option(
        False,
        "--show-context",
        help="Print retrieved and assembled context items",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw AnswerResult as JSON",
    ),
) -> None:
    """Ask a question grounded in the knowledge repository with verifiable citations.

    Note: This CLI is a trusted development tool allowing manual specification of hub and
    user tags. In production, the API (Phase 4) derives identity, org_id, and ACL tags
    strictly from the verified auth token.
    """
    settings = get_settings()
    effective_org_id = org_id or settings.DEFAULT_ORG_ID
    tags_set = frozenset(user_tags) if user_tags else frozenset()

    async def _ask() -> None:
        async with session_scope() as session:
            # 1. Look up hub by name
            hub_repo = HubRepository(session)
            hub_model = await hub_repo.get_by_name(effective_org_id, hub)
            if hub_model is None:
                typer.secho(
                    f"Error: Hub '{hub}' not found in organization '{effective_org_id}'.",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)

            hub_id = hub_model.id

        # 2. Build pipeline components
        from kb.db.session import get_session_factory

        session_factory = get_session_factory()
        embedder = build_embedder(settings)
        reranker = build_reranker(settings)
        llm_protocol = build_llm(settings)
        llm_client = LLMClient(llm_protocol)

        hybrid_retriever = HybridRetriever(session_factory, embedder)
        rerank_stage = RerankStage(
            reranker,
            RerankConfig(
                enabled=not no_rerank and settings.RERANK_ENABLED,
                top_n=settings.RERANK_TOP_N,
            ),
        )
        retriever = Retriever(hybrid_retriever, rerank_stage)
        assembler = ContextAssembler(
            config=ContextConfig(
                budget_tokens=settings.CONTEXT_BUDGET_TOKENS,
                expand_parents=settings.CONTEXT_EXPAND_PARENTS,
            ),
            session_factory=session_factory,
        )
        generator = Generator(
            llm_client,
            prompt_name="answer",
            prompt_version="1.0.0",
            on_uncited=settings.ON_UNCITED,
            snippet_max_chars=settings.SNIPPET_MAX_CHARS,
        )
        rag_service = RagService(
            retriever=retriever,
            assembler=assembler,
            generator=generator,
            refusal_min_score=settings.REFUSAL_MIN_SCORE,
        )

        access = AccessContext(
            org_id=effective_org_id,
            hub_id=hub_id,
            user_tags=tags_set,
        )

        # 3. Optional context inspection before generation
        if show_context:
            candidates = await retriever.retrieve(question, access)
            bundle = await assembler.assemble(candidates, access)
            summary_line = (
                f"\n--- Assembled Context ({len(bundle.items)} chunks, "
                f"{bundle.total_tokens} tokens) ---"
            )
            typer.echo(summary_line)
            for idx, item in enumerate(bundle.items, start=1):
                typer.echo(
                    f"[{idx}] Chunk ID: {item.chunk_id} | Doc: {item.document_title} "
                    f"(Score: {item.final_score:.4f})"
                )
                typer.echo(
                    f"    Page: {item.page} | Section: {item.section} | URI: {item.source_uri}"
                )
                typer.echo(f"    Text: {item.text[:120]}...\n")

        # 4. Execute ask pipeline
        result = await rag_service.ask(question, access)

        # 5. Render output
        if json_output:
            typer.echo(result.model_dump_json(indent=2))
            return

        if result.refusal:
            typer.secho(f"\nRefusal ({result.refusal_reason}):", fg=typer.colors.YELLOW, bold=True)
            typer.echo(result.answer)
            return

        typer.secho("\nAnswer:", fg=typer.colors.GREEN, bold=True)
        typer.echo(result.answer)

        if result.citations:
            typer.secho("\nSources:", bold=True)
            for idx, cit in enumerate(result.citations, start=1):
                page_str = f"Page {cit.page}" if cit.page is not None else "Page N/A"
                sec_str = f"Section: {cit.section}" if cit.section else ""
                loc_str = f" ({page_str}, {sec_str})" if sec_str else f" ({page_str})"
                trunc_suffix = "..." if cit.snippet_truncated else ""
                typer.echo(f"[{idx}] {cit.document_title}{loc_str} - {cit.source_uri}")
                typer.echo(f'    "{cit.snippet}{trunc_suffix}"\n')

    asyncio.run(_ask())


if __name__ == "__main__":
    app()
