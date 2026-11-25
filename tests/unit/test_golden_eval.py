"""Unit tests for Phase 3.1 golden dataset schema, I/O, generation, and review CLI."""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from kb.cli import app
from kb.domain.evaluation import (
    GoldenItem,
    GoldenType,
    Provenance,
    RestrictionKind,
)
from kb.evaluation.corpus_plan import load_corpus_plan
from kb.evaluation.generate import (
    from_handwritten,
    from_manifest,
    merge_golden_candidates,
)
from kb.evaluation.golden import (
    GoldenFormatError,
    golden_file_hash,
    load_golden,
    span_in_text,
    write_golden,
)


def _valid_factual(item_id: str = "q-0001", question: str = "What is X?") -> GoldenItem:
    return GoldenItem(
        id=item_id,
        question=question,
        reference_answer="X is 42.",
        type=GoldenType.factual,
        source_uris=["doc.pdf"],
        evidence_spans=["X is 42."],
        provenance=Provenance.handwritten,
    )


def test_schema_type_validation_rules() -> None:
    """Each GoldenType enforces its specific validation rules with clear errors."""
    # 1. Invalid id pattern
    with pytest.raises(ValidationError):
        _valid_factual(item_id="bad-1")

    # 2. factual / table require >=1 source_uri, >=1 evidence_span, non-empty answer
    with pytest.raises(ValidationError, match="source_uri"):
        GoldenItem(
            id="q-0001",
            question="Q?",
            reference_answer="Ans",
            type=GoldenType.factual,
            source_uris=[],
            evidence_spans=["span"],
            provenance=Provenance.manifest,
        )
    with pytest.raises(ValidationError, match="evidence_span"):
        GoldenItem(
            id="q-0002",
            question="Q?",
            reference_answer="Ans",
            type=GoldenType.table,
            source_uris=["t.csv"],
            evidence_spans=[],
            provenance=Provenance.manifest,
        )
    with pytest.raises(ValidationError, match="reference_answer"):
        GoldenItem(
            id="q-0003",
            question="Q?",
            reference_answer="   ",
            type=GoldenType.factual,
            source_uris=["d.pdf"],
            evidence_spans=["span"],
            provenance=Provenance.manifest,
        )

    # 3. multi_hop requires >=2 distinct source_uris and >=2 evidence_spans
    with pytest.raises(ValidationError, match="different documents"):
        GoldenItem(
            id="q-0004",
            question="Q?",
            reference_answer="Ans",
            type=GoldenType.multi_hop,
            source_uris=["same.pdf", "same.pdf"],
            evidence_spans=["s1", "s2"],
            provenance=Provenance.handwritten,
        )
    with pytest.raises(ValidationError, match="2 non-empty evidence_spans"):
        GoldenItem(
            id="q-0005",
            question="Q?",
            reference_answer="Ans",
            type=GoldenType.multi_hop,
            source_uris=["a.pdf", "b.pdf"],
            evidence_spans=["only_one"],
            provenance=Provenance.handwritten,
        )

    # 4. unanswerable requires empty source_uris, evidence_spans, reference_answer
    unans = GoldenItem(
        id="q-0006",
        question="What is Mars capital?",
        reference_answer="",
        type=GoldenType.unanswerable,
        provenance=Provenance.manifest,
    )
    assert unans.expected_refusal is True
    with pytest.raises(ValidationError, match="empty source_uris"):
        GoldenItem(
            id="q-0007",
            question="Q?",
            reference_answer="",
            type=GoldenType.unanswerable,
            source_uris=["a.pdf"],
            provenance=Provenance.manifest,
        )
    with pytest.raises(ValidationError, match="empty reference_answer"):
        GoldenItem(
            id="q-0008",
            question="Q?",
            reference_answer="Should be empty",
            type=GoldenType.unanswerable,
            provenance=Provenance.manifest,
        )

    # 5. access_restricted requires restriction_kind, >=1 source_uri, >=1 span, empty answer
    acc = GoldenItem(
        id="q-0009",
        question="Secret revenue?",
        reference_answer="",
        type=GoldenType.access_restricted,
        source_uris=["fin.xlsx"],
        evidence_spans=["Revenue: $28.4M"],
        restriction_kind=RestrictionKind.acl_tag,
        user_tags=[],
        tags=["acl:finance"],
        provenance=Provenance.manifest,
    )
    assert acc.expected_refusal is True
    with pytest.raises(ValidationError, match="restriction_kind"):
        GoldenItem(
            id="q-0010",
            question="Q?",
            reference_answer="",
            type=GoldenType.access_restricted,
            source_uris=["fin.xlsx"],
            evidence_spans=["span"],
            provenance=Provenance.manifest,
        )
    with pytest.raises(ValidationError, match="must not grant access"):
        GoldenItem(
            id="q-0011",
            question="Q?",
            reference_answer="",
            type=GoldenType.access_restricted,
            source_uris=["fin.xlsx"],
            evidence_spans=["span"],
            restriction_kind=RestrictionKind.acl_tag,
            user_tags=["finance"],
            tags=["acl:finance"],
            provenance=Provenance.manifest,
        )


def test_load_golden_collects_all_errors_with_line_numbers(tmp_path: Path) -> None:
    """GoldenFormatError lists EVERY line error, duplicate id, and duplicate normalized question."""
    path = tmp_path / "bad.jsonl"
    item1 = _valid_factual("q-0001", "What is the gift limit?").model_dump_json()
    dup_id = _valid_factual("q-0001", "Different question?").model_dump_json()
    dup_q = _valid_factual("q-0002", "  what   is the GIFT limit? ").model_dump_json()
    bad_schema = json.dumps({"id": "bad-id", "question": ""})
    lines = [
        item1,
        "{not valid json",
        bad_schema,
        dup_id,
        dup_q,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(GoldenFormatError) as exc_info:
        load_golden(path)

    errors = exc_info.value.errors
    line_nums = [ln for ln, _ in errors]
    assert line_nums == [2, 3, 4, 5]
    assert "Line 2:" in str(exc_info.value)
    assert "Line 4:" in str(exc_info.value)
    assert "Line 5:" in str(exc_info.value)


def test_load_golden_reviewed_only_and_canonical_write_hash(tmp_path: Path) -> None:
    """reviewed_only filters unreviewed items; write_golden is canonical and hash tracks changes."""
    path = tmp_path / "golden.jsonl"
    item2 = _valid_factual("q-0002", "Second question?").model_copy(update={"reviewed": True})
    item1 = _valid_factual("q-0001", "First question?").model_copy(update={"reviewed": False})

    write_golden(path, [item2, item1])
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    # Sorted by id: q-0001 before q-0002
    assert json.loads(raw_lines[0])["id"] == "q-0001"
    assert json.loads(raw_lines[1])["id"] == "q-0002"

    reviewed = load_golden(path, reviewed_only=True)
    assert [i.id for i in reviewed] == ["q-0002"]

    h1 = golden_file_hash(path)
    write_golden(path, [item1, item2])  # reversed input order -> identical canonical output
    h2 = golden_file_hash(path)
    assert h1 == h2

    item2_mod = item2.model_copy(update={"reference_answer": "Changed answer."})
    write_golden(path, [item1, item2_mod])
    h3 = golden_file_hash(path)
    assert h3 != h1


def test_span_in_text_normalization() -> None:
    """span_in_text normalizes case and whitespace."""
    text = "The   enterprise standard\nfor PostgreSQL   is 50 connections."
    assert span_in_text("enterprise standard for postgresql", text) is True
    assert span_in_text("not present", text) is False


def test_corpus_plan_and_manifest_generation() -> None:
    """Corpus plan validates and generated items meet target composition and controls."""
    plan = load_corpus_plan(Path("eval/golden/corpus_plan.json"))
    assignments = plan.resolve_files(Path("data/raw/sample"))
    assert len(assignments) >= 36

    hubs = Counter(g.hub for _, g in assignments)
    scopes = Counter(g.scope.value for _, g in assignments)
    assert hubs["other"] >= 4
    assert scopes["global"] >= 2

    manifest_items = from_manifest(
        Path("data/raw/sample/manifest.json"), plan, root=Path("data/raw/sample")
    )
    hw_items = from_handwritten(Path("eval/golden/handwritten.jsonl"))
    merged = merge_golden_candidates([], manifest_items + hw_items)

    assert len(merged) >= 64
    counts = Counter(i.type for i in merged)
    assert counts[GoldenType.factual] >= 24
    assert counts[GoldenType.table] >= 12
    assert counts[GoldenType.multi_hop] >= 10
    assert counts[GoldenType.unanswerable] >= 10
    assert counts[GoldenType.access_restricted] >= 8

    # Every access_restricted item has a control
    restricted_ids = {i.id for i in merged if i.type == GoldenType.access_restricted}
    controlled_ids = {i.control_for for i in merged if i.control_for is not None}
    assert restricted_ids == controlled_ids

    # Re-running merge never reuses IDs or alters reviewed items
    reviewed_first = merged[0].model_copy(
        update={
            "reviewed": True,
            "reviewed_by": "Alice",
            "reviewed_at": datetime.now(UTC),
            "question": "Custom reviewed question?",
        }
    )
    re_merged = merge_golden_candidates([reviewed_first], manifest_items + hw_items)
    assert re_merged[0] == reviewed_first
    assert len({i.id for i in re_merged}) == len(re_merged)


def test_review_cli_accept_edit_reject_skip_and_bulk_rules(tmp_path: Path) -> None:
    """Interactive review and bulk-accept CLI rules behave as specified."""
    golden_path = tmp_path / "golden.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    items = [
        _valid_factual("q-0001", "Question 1?").model_copy(
            update={"provenance": Provenance.manifest}
        ),
        _valid_factual("q-0002", "Question 2?").model_copy(update={"provenance": Provenance.llm}),
        _valid_factual("q-0003", "Question 3?").model_copy(
            update={"provenance": Provenance.handwritten}
        ),
        _valid_factual("q-0004", "Question 4?").model_copy(
            update={"provenance": Provenance.handwritten}
        ),
    ]
    write_golden(golden_path, items)
    runner = CliRunner()

    # 1. Bulk accept refuses llm provenance
    res_llm = runner.invoke(
        app,
        [
            "eval",
            "golden",
            "review",
            "--golden",
            str(golden_path),
            "--accept-provenance",
            "llm",
            "--reviewer",
            "Tester",
            "--yes",
        ],
    )
    assert res_llm.exit_code != 0
    assert "never be bulk-accepted" in res_llm.output

    # 2. Bulk accept refuses without --yes
    res_no_yes = runner.invoke(
        app,
        [
            "eval",
            "golden",
            "review",
            "--golden",
            str(golden_path),
            "--accept-provenance",
            "manifest",
            "--reviewer",
            "Tester",
        ],
    )
    assert res_no_yes.exit_code != 0
    assert "--yes" in res_no_yes.output

    # 3. Interactive review: accept q-0001, edit q-0002, reject q-0003, skip q-0004
    scripted_input = (
        "\n".join(
            [
                "a",  # accept q-0001
                "e",  # edit q-0002
                "Edited Question 2?",
                "Edited Answer 2.",
                "r",  # reject q-0003
                "too vague",
                "s",  # skip q-0004
            ]
        )
        + "\n"
    )
    res_int = runner.invoke(
        app,
        [
            "eval",
            "golden",
            "review",
            "--golden",
            str(golden_path),
            "--reviewer",
            "Tester",
            "--rejected-out",
            str(rejected_path),
        ],
        input=scripted_input,
    )
    assert res_int.exit_code == 0, res_int.output

    final_items = {i.id: i for i in load_golden(golden_path)}
    assert set(final_items.keys()) == {"q-0001", "q-0002", "q-0004"}
    assert final_items["q-0001"].reviewed is True
    assert final_items["q-0001"].reviewed_by == "Tester"
    assert final_items["q-0002"].question == "Edited Question 2?"
    assert final_items["q-0002"].reviewed is True
    assert final_items["q-0004"].reviewed is False

    rejected_lines = rejected_path.read_text(encoding="utf-8").splitlines()
    assert len(rejected_lines) == 1
    assert json.loads(rejected_lines[0])["id"] == "q-0003"
    assert json.loads(rejected_lines[0])["rejection_reason"] == "too vague"
