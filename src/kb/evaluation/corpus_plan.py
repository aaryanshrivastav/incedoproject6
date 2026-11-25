"""Corpus plan schema, loader, and preflight DB verification."""

import json
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kb.db.models import ChunkModel, DocumentModel, HubDocumentModel, HubModel
from kb.domain.evaluation import GoldenItem, GoldenType, RestrictionKind
from kb.domain.models import DocumentScope
from kb.evaluation.golden import span_in_text
from kb.ingestion.normalize import normalize_text


class CorpusPlanGroup(BaseModel):
    """Single ingestion group rule mapping file globs to hub, scope, and ACL tags."""

    glob: str = Field(min_length=1, description="Glob pattern relative to corpus root")
    hub: str = Field(default="default", min_length=1, description="Target hub name")
    scope: DocumentScope = Field(default=DocumentScope.HUB, description="Document scope")
    acl_tags: list[str] = Field(default_factory=list, description="ACL tags applied on ingest")


class CorpusPlan(BaseModel):
    """Ordered list of ingestion groups for setting up an evaluation corpus."""

    groups: list[CorpusPlanGroup] = Field(min_length=1)

    def resolve_files(self, root: Path) -> list[tuple[Path, CorpusPlanGroup]]:
        """Resolve each corpus file to its matching plan group (first match wins)."""
        assigned: dict[Path, CorpusPlanGroup] = {}
        for group in self.groups:
            for matched in sorted(root.glob(group.glob)):
                if matched.is_file() and matched.name != "manifest.json":
                    resolved = matched.resolve()
                    if resolved not in assigned:
                        assigned[resolved] = group
        return sorted(assigned.items(), key=lambda pair: str(pair[0]))

    def expected_for_uri(self, source_uri: str, root: Path | None = None) -> CorpusPlanGroup | None:
        """Find the plan group matching a relative source_uri."""
        if root is not None:
            for path, group in self.resolve_files(root):
                if path.relative_to(root.resolve()).as_posix() == source_uri:
                    return group
        for group in self.groups:
            if Path(source_uri).match(group.glob):
                return group
        return None


def load_corpus_plan(path: Path) -> CorpusPlan:
    """Load and validate a corpus plan JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Corpus plan not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        if isinstance(raw, list):
            return CorpusPlan(groups=[CorpusPlanGroup.model_validate(item) for item in raw])
        return CorpusPlan.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid corpus plan in {path}: {exc}") from exc


async def check_corpus_against_golden(
    session: AsyncSession,
    org_id: UUID,
    items: list[GoldenItem],
    plan: CorpusPlan | None = None,
    root: Path | None = None,
) -> list[str]:
    """Run preflight checks verifying DB documents, ACL/hub metadata, spans, and unanswerables."""
    errors: list[str] = []

    # Load all active chunks for unanswerable token checks
    all_chunks_stmt = (
        select(ChunkModel.text)
        .join(DocumentModel, DocumentModel.id == ChunkModel.document_id)
        .where(
            DocumentModel.org_id == org_id,
            DocumentModel.status == "active",
            ChunkModel.is_active.is_(True),
        )
    )
    all_active_texts = [
        normalize_text(str(row[0])).casefold()
        for row in (await session.execute(all_chunks_stmt)).all()
    ]

    for item in items:
        if item.type == GoldenType.unanswerable:
            tokens = [t.removeprefix("token:").strip() for t in item.tags if t.strip()]
            for tok in tokens:
                tok_norm = normalize_text(tok).casefold()
                if any(tok_norm in chunk_txt for chunk_txt in all_active_texts):
                    errors.append(
                        f"[{item.id}] Unanswerable token '{tok}' unexpectedly found in corpus"
                    )
            continue

        # Collect active chunks for each referenced source_uri
        doc_chunks_map: dict[str, list[str]] = {}
        for uri in item.source_uris:
            doc_stmt = select(DocumentModel).where(
                DocumentModel.org_id == org_id,
                DocumentModel.source_uri == uri,
                DocumentModel.status == "active",
            )
            doc = (await session.execute(doc_stmt)).scalar_one_or_none()
            if doc is None:
                errors.append(f"[{item.id}] Missing active document for source_uri '{uri}'")
                continue

            # Fetch hub names associated with this document
            hubs_stmt = (
                select(HubModel.name)
                .join(HubDocumentModel, HubDocumentModel.hub_id == HubModel.id)
                .where(HubDocumentModel.document_id == doc.id)
            )
            doc_hubs = {str(r[0]) for r in (await session.execute(hubs_stmt)).all()}

            # Verify expected plan metadata if plan is supplied
            if plan is not None:
                expected = plan.expected_for_uri(uri, root=root)
                if expected is not None:
                    if doc.scope != expected.scope.value:
                        errors.append(
                            f"[{item.id}] '{uri}' has scope '{doc.scope}', "
                            f"expected '{expected.scope.value}'"
                        )
                    if set(doc.acl_tags) != set(expected.acl_tags):
                        errors.append(
                            f"[{item.id}] '{uri}' has acl_tags {list(doc.acl_tags)}, "
                            f"expected {expected.acl_tags}"
                        )
                    if expected.hub not in doc_hubs:
                        errors.append(
                            f"[{item.id}] '{uri}' missing expected hub membership '{expected.hub}'"
                        )

            # Verify access_restricted invariants against real DB metadata
            if item.type == GoldenType.access_restricted:
                if item.restriction_kind == RestrictionKind.acl_tag:
                    if not doc.acl_tags or set(doc.acl_tags).intersection(item.user_tags):
                        errors.append(
                            f"[{item.id}] acl_tag restriction invalid: "
                            f"doc acl_tags={doc.acl_tags}, user_tags={item.user_tags}"
                        )
                elif item.restriction_kind == RestrictionKind.other_hub:
                    if doc.scope == DocumentScope.GLOBAL.value or item.hub in doc_hubs:
                        errors.append(
                            f"[{item.id}] other_hub restriction invalid: doc is visible from "
                            f"hub '{item.hub}' (scope={doc.scope}, hubs={sorted(doc_hubs)})"
                        )

            chunks_stmt = (
                select(ChunkModel.text)
                .where(ChunkModel.document_id == doc.id, ChunkModel.is_active.is_(True))
                .order_by(ChunkModel.ordinal)
            )
            doc_chunks_map[uri] = [str(r[0]) for r in (await session.execute(chunks_stmt)).all()]

        # Check each evidence span appears in a single active chunk of at least one referenced doc
        all_doc_chunks = [c for chunks in doc_chunks_map.values() for c in chunks]
        if not all_doc_chunks:
            continue

        for span in item.evidence_spans:
            found_in_single_chunk = any(span_in_text(span, c) for c in all_doc_chunks)
            if found_in_single_chunk:
                continue

            # Check if span crosses chunk boundary inside any referenced document
            crosses_boundary = any(
                span_in_text(span, " ".join(chunks)) or span_in_text(span, "\n".join(chunks))
                for chunks in doc_chunks_map.values()
            )
            if crosses_boundary:
                errors.append(f"[{item.id}] span crosses chunk boundary: '{span[:80]}'")
            else:
                errors.append(f"[{item.id}] span not found: '{span[:80]}'")

    return errors
