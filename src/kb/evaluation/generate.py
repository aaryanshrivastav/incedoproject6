"""Golden dataset candidate generators (manifest, handwritten, LLM) and builder."""

import json
import random
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kb.db.models import ChunkModel, DocumentModel, HubDocumentModel, HubModel
from kb.domain.evaluation import GoldenItem, GoldenType, Provenance, RestrictionKind
from kb.generation.citations import parse_model_output
from kb.generation.prompts import load_prompt
from kb.ingestion.normalize import normalize_text
from kb.llm.client import LLMClient


def _next_id(counter: int) -> str:
    return f"q-{counter:04d}"


def from_manifest(
    manifest_path: Path,
    plan: Any,
    *,
    root: Path | None = None,
) -> list[GoldenItem]:
    """Generate unreviewed GoldenItem candidates from an enriched manifest.json and corpus plan."""
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    items: list[GoldenItem] = []
    counter = 1

    # Track restricted items so each gets an authorized control
    for filename, entry in sorted(manifest.items()):
        if filename in ("unanswerable_questions", "multi_hop") or not isinstance(entry, dict):
            continue
        question = str(entry.get("question", "")).strip()
        answer = str(entry.get("answer", "")).strip()
        span = str(entry.get("evidence_span", "")).strip()
        kind_str = str(entry.get("kind", "factual")).strip().lower()
        if not (question and answer and span):
            continue

        item_type = GoldenType.table if kind_str == "table" else GoldenType.factual
        source_uri = str(entry.get("source", filename))
        group = plan.expected_for_uri(source_uri, root=root)
        doc_hub = group.hub if group is not None else "default"
        doc_acl = list(group.acl_tags) if group is not None else []
        acl_hint = str(entry.get("acl_hint", "")).strip()

        # Check if this document is restricted (finance ACL or other hub)
        if acl_hint or doc_acl:
            required_tag = acl_hint or doc_acl[0]
            restricted_id = _next_id(counter)
            counter += 1
            items.append(
                GoldenItem(
                    id=restricted_id,
                    question=question,
                    reference_answer="",
                    type=GoldenType.access_restricted,
                    hub=doc_hub,
                    user_tags=[],
                    source_uris=[source_uri],
                    evidence_spans=[span],
                    restriction_kind=RestrictionKind.acl_tag,
                    provenance=Provenance.manifest,
                    reviewed=False,
                    notes=f"ACL negative test: requires '{required_tag}' tag",
                    tags=[f"acl:{required_tag}"],
                )
            )
            # Authorized control item
            control_id = _next_id(counter)
            counter += 1
            items.append(
                GoldenItem(
                    id=control_id,
                    question=question,
                    reference_answer=answer,
                    type=item_type,
                    hub=doc_hub,
                    user_tags=[required_tag],
                    source_uris=[source_uri],
                    evidence_spans=[span],
                    control_for=restricted_id,
                    provenance=Provenance.manifest,
                    reviewed=False,
                    notes=f"Authorized control for {restricted_id}",
                    tags=[f"acl:{required_tag}"],
                )
            )
        elif doc_hub != "default":
            restricted_id = _next_id(counter)
            counter += 1
            items.append(
                GoldenItem(
                    id=restricted_id,
                    question=question,
                    reference_answer="",
                    type=GoldenType.access_restricted,
                    hub="default",
                    user_tags=[],
                    source_uris=[source_uri],
                    evidence_spans=[span],
                    restriction_kind=RestrictionKind.other_hub,
                    provenance=Provenance.manifest,
                    reviewed=False,
                    notes=f"Cross-hub negative test: doc lives in hub '{doc_hub}'",
                    tags=[f"target_hub:{doc_hub}"],
                )
            )
            control_id = _next_id(counter)
            counter += 1
            items.append(
                GoldenItem(
                    id=control_id,
                    question=question,
                    reference_answer=answer,
                    type=item_type,
                    hub=doc_hub,
                    user_tags=[],
                    source_uris=[source_uri],
                    evidence_spans=[span],
                    control_for=restricted_id,
                    provenance=Provenance.manifest,
                    reviewed=False,
                    notes=f"Authorized control for {restricted_id}",
                    tags=[f"target_hub:{doc_hub}"],
                )
            )
        else:
            items.append(
                GoldenItem(
                    id=_next_id(counter),
                    question=question,
                    reference_answer=answer,
                    type=item_type,
                    hub="default",
                    user_tags=[],
                    source_uris=[source_uri],
                    evidence_spans=[span],
                    provenance=Provenance.manifest,
                    reviewed=False,
                    notes=f"Manifest {item_type.value} from {source_uri}",
                )
            )
            counter += 1

        # Add extra table rows if present so table >= 12
        extra_rows = entry.get("extra_rows")
        if isinstance(extra_rows, list) and not (acl_hint or doc_acl) and doc_hub == "default":
            for extra in extra_rows:
                if not isinstance(extra, dict):
                    continue
                eq = str(extra.get("question", "")).strip()
                ea = str(extra.get("answer", "")).strip()
                es = str(extra.get("evidence_span", "")).strip()
                if eq and ea and es:
                    items.append(
                        GoldenItem(
                            id=_next_id(counter),
                            question=eq,
                            reference_answer=ea,
                            type=GoldenType.table,
                            hub="default",
                            user_tags=[],
                            source_uris=[source_uri],
                            evidence_spans=[es],
                            provenance=Provenance.manifest,
                            reviewed=False,
                            notes=f"Manifest table row from {source_uri}",
                        )
                    )
                    counter += 1

    # Unanswerable items from manifest
    for u_entry in manifest.get("unanswerable_questions", []):
        if not isinstance(u_entry, dict):
            continue
        uq = str(u_entry.get("question", "")).strip()
        if not uq:
            continue
        tokens = [str(t) for t in u_entry.get("distinctive_tokens", []) if str(t).strip()]
        items.append(
            GoldenItem(
                id=_next_id(counter),
                question=uq,
                reference_answer="",
                type=GoldenType.unanswerable,
                hub="default",
                user_tags=[],
                source_uris=[],
                evidence_spans=[],
                provenance=Provenance.manifest,
                reviewed=False,
                notes=str(u_entry.get("notes", "Unanswerable out-of-corpus query")),
                tags=[f"token:{t}" for t in tokens],
            )
        )
        counter += 1

    return items


def from_handwritten(path: Path) -> list[GoldenItem]:
    """Load handwritten GoldenItems from JSONL, ensuring provenance=handwritten."""
    from kb.evaluation.golden import load_golden

    items = load_golden(path, reviewed_only=False)
    return [item.model_copy(update={"provenance": Provenance.handwritten}) for item in items]


async def from_chunks_llm(
    session: AsyncSession,
    llm_client: LLMClient,
    n: int,
    seed: int = 7,
    org_id: UUID | None = None,
) -> list[GoldenItem]:
    """Deterministically sample active chunks and generate candidate items via LLMClient.

    Enforces I3 (delimited untrusted chunk text), I7 (LLMClient wrapper), and I9 (file prompt).
    Discards any output whose evidence_span is not an exact substring of the sampled chunk.
    """
    if n <= 0:
        return []

    stmt = (
        select(ChunkModel.text, DocumentModel.source_uri, HubModel.name)
        .join(DocumentModel, DocumentModel.id == ChunkModel.document_id)
        .join(HubDocumentModel, HubDocumentModel.document_id == DocumentModel.id)
        .join(HubModel, HubModel.id == HubDocumentModel.hub_id)
        .where(
            DocumentModel.status == "active",
            ChunkModel.is_active.is_(True),
        )
    )
    if org_id is not None:
        stmt = stmt.where(DocumentModel.org_id == org_id)
    stmt = stmt.order_by(DocumentModel.source_uri, ChunkModel.ordinal)
    rows = (await session.execute(stmt)).all()
    if not rows:
        return []

    rng = random.Random(seed)
    sampled = rng.sample(list(rows), k=min(n, len(rows)))

    _, system_prompt = load_prompt("golden_generate", "1.0.0")
    results: list[GoldenItem] = []

    for idx, (chunk_text, source_uri, hub_name) in enumerate(sampled, start=1):
        chunk_str = str(chunk_text)
        # Escape closing chunk tags inside untrusted data (I3)
        safe_chunk = chunk_str.replace("</chunk>", "&lt;/chunk&gt;")
        user_prompt = (
            f"Source URI: {source_uri}\n"
            f"<chunk>\n{safe_chunk}\n</chunk>\n\n"
            "Generate one factual JSON item grounded strictly in the chunk above."
        )
        res = await llm_client.generate(user_prompt, system_prompt=system_prompt, temperature=0.0)
        try:
            parsed = parse_model_output(res.response.content)
        except Exception:
            continue

        question = str(parsed.get("question", "")).strip()
        ref_answer = str(parsed.get("reference_answer", "")).strip()
        span = str(parsed.get("evidence_span", "")).strip()
        if not (question and ref_answer and span):
            continue

        # Strictly discard if evidence_span is not an exact substring of the sampled chunk
        if span not in chunk_str:
            continue

        results.append(
            GoldenItem(
                id=_next_id(9000 + idx),
                question=question,
                reference_answer=ref_answer,
                type=GoldenType.factual,
                hub=str(hub_name),
                user_tags=[],
                source_uris=[str(source_uri)],
                evidence_spans=[span],
                provenance=Provenance.llm,
                reviewed=False,
                notes="LLM-generated candidate from active chunk",
            )
        )

    return results


def merge_golden_candidates(
    existing_items: list[GoldenItem],
    candidates: list[GoldenItem],
) -> list[GoldenItem]:
    """Merge new candidates into existing items without altering reviewed items or reusing IDs."""
    # Preserve all existing reviewed items untouched
    kept: list[GoldenItem] = list(existing_items)
    used_ids: set[str] = {item.id for item in kept}
    seen_questions: set[str] = {
        normalize_text(item.question).casefold() for item in kept if item.control_for is None
    }

    max_num = 0
    for item_id in used_ids:
        try:
            max_num = max(max_num, int(item_id.split("-")[1]))
        except (IndexError, ValueError):
            pass

    next_num = max_num + 1
    id_remap: dict[str, str] = {}

    for cand in candidates:
        norm_q = normalize_text(cand.question).casefold()
        if cand.control_for is None and norm_q in seen_questions:
            continue

        new_id = f"q-{next_num:04d}"
        next_num += 1
        id_remap[cand.id] = new_id

        updated_control = (
            id_remap.get(cand.control_for, cand.control_for)
            if cand.control_for is not None
            else None
        )
        merged = cand.model_copy(update={"id": new_id, "control_for": updated_control})
        kept.append(merged)
        used_ids.add(new_id)
        if cand.control_for is None:
            seen_questions.add(norm_q)

    return kept
