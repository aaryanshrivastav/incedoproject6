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
from kb.ingestion.pipeline import IngestionPipeline, IngestRequest
from kb.llm.factory import build_embedder

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


if __name__ == "__main__":
    app()
