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

eval_app = typer.Typer(
    name="eval",
    help="Evaluation dataset, corpus setup, and benchmark commands",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")

golden_app = typer.Typer(
    name="golden",
    help="Build, review, and validate golden evaluation datasets",
    no_args_is_help=True,
)
eval_app.add_typer(golden_app, name="golden")


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


@eval_app.command(name="setup")
def eval_setup(
    plan: Path = typer.Option(
        Path("eval/golden/corpus_plan.json"),
        "--plan",
        help="Path to corpus plan JSON file",
    ),
    root: Path = typer.Option(
        Path("data/raw/sample"),
        "--root",
        help="Corpus root directory",
    ),
    org_id: UUID | None = typer.Option(
        None, "--org-id", help="Organization UUID (defaults to DEFAULT_ORG_ID)"
    ),
) -> None:
    """Idempotently ingest evaluation corpus files according to a corpus plan."""
    from kb.db.session import get_session_factory
    from kb.evaluation.corpus_plan import load_corpus_plan

    settings = get_settings()
    effective_org_id = org_id or settings.DEFAULT_ORG_ID
    corpus_plan = load_corpus_plan(plan)
    assignments = corpus_plan.resolve_files(root)

    if not assignments:
        typer.secho(f"No files matched plan '{plan}' under '{root}'.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    embedder = build_embedder(settings)
    session_factory = get_session_factory()
    pipeline = IngestionPipeline(embedder=embedder, session_factory=session_factory)

    async def _setup() -> None:
        total_created = 0
        total_updated = 0
        total_skipped = 0
        total_meta_updated = 0
        total_failed = 0
        total_chunks = 0

        for file_path, group in assignments:
            req = IngestRequest(
                org_id=effective_org_id,
                hub_name=group.hub,
                scope=group.scope,
                acl_tags=list(group.acl_tags),
                source_root=root,
            )
            res = await pipeline.ingest_file(file_path, req)
            total_chunks += res.chunks_created
            if res.status.value == "created":
                total_created += 1
            elif res.status.value == "updated":
                total_updated += 1
            elif res.status.value == "skipped":
                total_skipped += 1
            elif res.status.value == "metadata_updated":
                total_meta_updated += 1
            elif res.status.value == "failed":
                total_failed += 1

        typer.echo("\n--- Eval Corpus Setup Summary ---")
        typer.echo(f"  Files Processed  : {len(assignments)}")
        typer.echo(f"  Created          : {total_created}")
        typer.echo(f"  Updated          : {total_updated}")
        typer.echo(f"  Skipped          : {total_skipped}")
        typer.echo(f"  Metadata Updated : {total_meta_updated}")
        typer.echo(f"  Failed           : {total_failed}")
        typer.echo(f"  New Chunks       : {total_chunks}")

    asyncio.run(_setup())


@eval_app.command(name="check-corpus")
def eval_check_corpus(
    golden: Path = typer.Option(
        Path("eval/golden/golden.jsonl"),
        "--golden",
        help="Path to golden JSONL file",
    ),
    root: Path = typer.Option(
        Path("data/raw/sample"),
        "--root",
        help="Corpus root directory",
    ),
    plan: Path | None = typer.Option(
        Path("eval/golden/corpus_plan.json"),
        "--plan",
        help="Optional corpus plan JSON for verifying ACL/hub metadata",
    ),
    org_id: UUID | None = typer.Option(
        None, "--org-id", help="Organization UUID (defaults to DEFAULT_ORG_ID)"
    ),
) -> None:
    """Preflight check verifying documents, ACL/hub metadata, spans, and unanswerable tokens."""
    from kb.evaluation.corpus_plan import check_corpus_against_golden, load_corpus_plan
    from kb.evaluation.golden import GoldenFormatError, load_golden

    settings = get_settings()
    effective_org_id = org_id or settings.DEFAULT_ORG_ID

    try:
        items = load_golden(golden, reviewed_only=False)
    except (FileNotFoundError, GoldenFormatError) as exc:
        typer.secho(f"Golden Format Error:\n{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    loaded_plan = load_corpus_plan(plan) if plan is not None and plan.exists() else None

    async def _check() -> list[str]:
        async with session_scope() as session:
            return await check_corpus_against_golden(
                session=session,
                org_id=effective_org_id,
                items=items,
                plan=loaded_plan,
                root=root,
            )

    problems = asyncio.run(_check())
    if problems:
        typer.secho(f"Corpus Preflight FAILED ({len(problems)} problem(s)):", fg=typer.colors.RED)
        for prob in problems:
            typer.echo(f"  - {prob}")
        raise typer.Exit(code=1)

    typer.secho(
        f"Corpus Preflight PASSED: all {len(items)} golden items verified against active corpus.",
        fg=typer.colors.GREEN,
        bold=True,
    )


@golden_app.command(name="build")
def golden_build(
    out: Path = typer.Option(
        Path("eval/golden/golden.jsonl"),
        "--out",
        help="Output golden JSONL path",
    ),
    manifest: Path = typer.Option(
        Path("data/raw/sample/manifest.json"),
        "--manifest",
        help="Path to enriched corpus manifest.json",
    ),
    plan: Path = typer.Option(
        Path("eval/golden/corpus_plan.json"),
        "--plan",
        help="Path to corpus_plan.json",
    ),
    handwritten: Path = typer.Option(
        Path("eval/golden/handwritten.jsonl"),
        "--handwritten",
        help="Path to handwritten.jsonl",
    ),
    root: Path = typer.Option(
        Path("data/raw/sample"),
        "--root",
        help="Corpus root directory",
    ),
    llm_count: int = typer.Option(
        0,
        "--llm-count",
        help="Number of active chunks to sample for LLM candidate generation (default: 0)",
    ),
    seed: int = typer.Option(7, "--seed", help="Random seed for LLM sampling"),
) -> None:
    """Build or update golden.jsonl from manifest, handwritten, and optional LLM candidates."""
    from kb.domain.evaluation import GoldenItem
    from kb.evaluation.corpus_plan import load_corpus_plan
    from kb.evaluation.generate import (
        from_chunks_llm,
        from_handwritten,
        from_manifest,
        merge_golden_candidates,
    )
    from kb.evaluation.golden import load_golden, write_golden

    corpus_plan = load_corpus_plan(plan)
    existing: list[GoldenItem] = []
    if out.exists():
        existing = load_golden(out, reviewed_only=False)

    candidates: list[GoldenItem] = []
    candidates.extend(from_manifest(manifest, corpus_plan, root=root))
    if handwritten.exists():
        candidates.extend(from_handwritten(handwritten))

    if llm_count > 0:
        settings = get_settings()
        llm_client = LLMClient(build_llm(settings))

        async def _gen_llm() -> list[GoldenItem]:
            async with session_scope() as session:
                return await from_chunks_llm(session, llm_client, n=llm_count, seed=seed)

        candidates.extend(asyncio.run(_gen_llm()))

    merged = merge_golden_candidates(existing, candidates)
    write_golden(out, merged)
    typer.secho(f"Built {len(merged)} golden items -> {out}", fg=typer.colors.GREEN, bold=True)


def _resolve_reviewer(reviewer: str | None) -> str:
    """Resolve reviewer identity from --reviewer or git config user.name."""
    import subprocess

    if reviewer and reviewer.strip():
        return reviewer.strip()
    proc = subprocess.run(
        ["git", "config", "user.name"],
        capture_output=True,
        text=True,
        check=False,
    )
    git_name = proc.stdout.strip()
    if git_name:
        return git_name
    typer.secho(
        "Error: --reviewer is required when git user.name is unset.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1)


@golden_app.command(name="review")
def golden_review(
    golden: Path = typer.Option(
        Path("eval/golden/golden.jsonl"),
        "--golden",
        help="Path to golden JSONL file",
    ),
    only_pending: bool = typer.Option(
        False, "--only-pending", help="Review only items with reviewed=False"
    ),
    item_type: str | None = typer.Option(None, "--type", help="Filter items by GoldenType"),
    reviewer: str | None = typer.Option(
        None, "--reviewer", help="Reviewer name (defaults to git user.name)"
    ),
    accept_provenance: str | None = typer.Option(
        None,
        "--accept-provenance",
        help="Bulk-accept items of this provenance (never allowed for 'llm')",
    ),
    yes: bool = typer.Option(False, "--yes", help="Confirm bulk acceptance"),
    rejected_out: Path = typer.Option(
        Path("eval/golden/rejected.jsonl"),
        "--rejected-out",
        help="JSONL path where rejected items are appended",
    ),
    org_id: UUID | None = typer.Option(
        None, "--org-id", help="Organization UUID (defaults to DEFAULT_ORG_ID)"
    ),
) -> None:
    """Interactively review golden items or bulk-accept verified manifest items."""
    from datetime import UTC, datetime

    from kb.domain.evaluation import GoldenItem, GoldenType, Provenance
    from kb.evaluation.corpus_plan import check_corpus_against_golden, load_corpus_plan
    from kb.evaluation.golden import load_golden, write_golden

    reviewer_name = _resolve_reviewer(reviewer)
    items = load_golden(golden, reviewed_only=False)

    if accept_provenance is not None:
        prov_val = accept_provenance.strip().lower()
        if prov_val == Provenance.llm.value:
            typer.secho(
                "Error: Items with provenance 'llm' can never be bulk-accepted.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        if not yes:
            typer.secho(
                "Error: Bulk acceptance requires --yes.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        try:
            target_prov = Provenance(prov_val)
        except ValueError as exc:
            typer.secho(f"Error: Unknown provenance '{prov_val}'.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

        matching = [item for item in items if item.provenance == target_prov]
        if not matching:
            typer.secho(
                f"No items found with provenance '{target_prov.value}'.", fg=typer.colors.YELLOW
            )
            return

        settings = get_settings()
        effective_org_id = org_id or settings.DEFAULT_ORG_ID
        plan_path = golden.parent / "corpus_plan.json"
        loaded_plan = load_corpus_plan(plan_path) if plan_path.exists() else None

        async def _verify_matching() -> list[str]:
            async with session_scope() as session:
                return await check_corpus_against_golden(
                    session=session,
                    org_id=effective_org_id,
                    items=matching,
                    plan=loaded_plan,
                )

        problems = asyncio.run(_verify_matching())
        if problems:
            typer.secho(
                f"Bulk accept aborted: check-corpus found {len(problems)} error(s):",
                fg=typer.colors.RED,
                err=True,
            )
            for prob in problems:
                typer.echo(f"  - {prob}", err=True)
            raise typer.Exit(code=1)

        now_utc = datetime.now(UTC)
        updated_items: list[GoldenItem] = []
        accepted_count = 0
        for item in items:
            if item.provenance == target_prov:
                updated_items.append(
                    item.model_copy(
                        update={
                            "reviewed": True,
                            "reviewed_by": reviewer_name,
                            "reviewed_at": now_utc,
                        }
                    )
                )
                accepted_count += 1
            else:
                updated_items.append(item)

        write_golden(golden, updated_items)
        typer.secho(
            f"Attestation: Reviewer '{reviewer_name}' attests that the {accepted_count} "
            f"planted facts with provenance '{target_prov.value}' are correct by construction.",
            fg=typer.colors.GREEN,
        )
        return

    # Interactive review mode
    target_type = GoldenType(item_type) if item_type else None
    remaining: list[GoldenItem] = []
    rejected_records: list[dict[str, object]] = []
    quit_requested = False

    for item in items:
        if quit_requested:
            remaining.append(item)
            continue
        if only_pending and item.reviewed:
            remaining.append(item)
            continue
        if target_type is not None and item.type != target_type:
            remaining.append(item)
            continue

        typer.echo(
            f"\n=== [{item.id}] type={item.type.value} provenance={item.provenance.value} ==="
        )
        typer.echo(f"Question         : {item.question}")
        typer.echo(f"Reference Answer : {item.reference_answer or '(empty)'}")
        typer.echo(f"Source URIs      : {item.source_uris}")
        for idx, sp in enumerate(item.evidence_spans, start=1):
            typer.echo(f"  Span {idx}: {sp}")

        action = typer.prompt("Action [a]ccept / [e]dit / [r]eject / [s]kip / [q]uit", default="s")
        act = action.strip().lower()

        if act in ("q", "quit"):
            quit_requested = True
            remaining.append(item)
        elif act in ("a", "accept"):
            remaining.append(
                item.model_copy(
                    update={
                        "reviewed": True,
                        "reviewed_by": reviewer_name,
                        "reviewed_at": datetime.now(UTC),
                    }
                )
            )
        elif act in ("e", "edit"):
            new_q = typer.prompt("New question (Enter to keep)", default=item.question)
            new_a = typer.prompt(
                "New reference answer (Enter to keep)", default=item.reference_answer
            )
            edited = GoldenItem.model_validate(
                {
                    **item.model_dump(mode="python"),
                    "question": new_q,
                    "reference_answer": new_a,
                    "reviewed": True,
                    "reviewed_by": reviewer_name,
                    "reviewed_at": datetime.now(UTC),
                }
            )
            remaining.append(edited)
        elif act in ("r", "reject"):
            reason = typer.prompt("Rejection reason", default="low quality")
            rec = item.model_dump(mode="json")
            rec["rejection_reason"] = reason
            rec["rejected_by"] = reviewer_name
            rec["rejected_at"] = datetime.now(UTC).isoformat()
            rejected_records.append(rec)
        else:
            remaining.append(item)

    write_golden(golden, remaining)
    if rejected_records:
        rejected_out.parent.mkdir(parents=True, exist_ok=True)
        with rejected_out.open("a", encoding="utf-8") as fh:
            for rec in rejected_records:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
    typer.secho(
        f"Review saved: {len(remaining)} kept in {golden}, {len(rejected_records)} rejected.",
        fg=typer.colors.GREEN,
    )


@golden_app.command(name="validate")
def golden_validate(
    golden: Path = typer.Option(
        Path("eval/golden/golden.jsonl"),
        "--golden",
        help="Path to golden JSONL file",
    ),
    reviewed_only: bool = typer.Option(
        False, "--reviewed-only", help="Count only reviewed=True items"
    ),
    min_items: int | None = typer.Option(None, "--min-items", help="Minimum total items required"),
    min_unanswerable_fraction: float | None = typer.Option(
        None,
        "--min-unanswerable-fraction",
        help="Minimum fraction of unanswerable items (e.g. 0.1)",
    ),
    require_types: str | None = typer.Option(
        None,
        "--require-types",
        help="Comma-separated list of required GoldenTypes",
    ),
) -> None:
    """Validate golden JSONL schema and optionally enforce composition targets."""
    from collections import Counter

    from kb.domain.evaluation import GoldenType
    from kb.evaluation.golden import GoldenFormatError, load_golden

    try:
        items = load_golden(golden, reviewed_only=reviewed_only)
    except (FileNotFoundError, GoldenFormatError) as exc:
        typer.secho(f"Validation Error:\n{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    type_counts = Counter(item.type.value for item in items)
    prov_counts = Counter(item.provenance.value for item in items)
    reviewed_count = sum(1 for item in items if item.reviewed)

    typer.secho(f"\nGolden Dataset Valid: {golden}", fg=typer.colors.GREEN, bold=True)
    typer.echo("+----------------------+-------+")
    typer.echo("| Metric / Category    | Count |")
    typer.echo("+----------------------+-------+")
    typer.echo(f"| Total Items          | {len(items):5d} |")
    typer.echo(f"| Reviewed Items       | {reviewed_count:5d} |")
    for t in GoldenType:
        typer.echo(f"| type:{t.value:<15} | {type_counts[t.value]:5d} |")
    for k, v in sorted(prov_counts.items()):
        typer.echo(f"| prov:{k:<15} | {v:5d} |")
    typer.echo("+----------------------+-------+")

    violations: list[str] = []
    if min_items is not None and len(items) < min_items:
        violations.append(f"Expected >= {min_items} items, found {len(items)}.")
    if min_unanswerable_fraction is not None:
        frac = (type_counts[GoldenType.unanswerable.value] / len(items)) if items else 0.0
        if frac < min_unanswerable_fraction:
            violations.append(
                f"Unanswerable fraction {frac:.3f} < required {min_unanswerable_fraction:.3f}."
            )
    if require_types:
        for req_t in [s.strip() for s in require_types.split(",") if s.strip()]:
            if type_counts[req_t] == 0:
                violations.append(f"Required type '{req_t}' has 0 items.")

    if violations:
        for violation in violations:
            typer.secho(f"Violation: {violation}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
