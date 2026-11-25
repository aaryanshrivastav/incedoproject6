"""Golden dataset JSONL loading, canonical writing, hashing, and text matching."""

import hashlib
import json
import re
from pathlib import Path

from pydantic import ValidationError

from kb.domain.evaluation import GoldenItem
from kb.ingestion.normalize import normalize_text

_WS_RE = re.compile(r"\s+")


class GoldenFormatError(Exception):
    """Raised when a golden JSONL file has one or more format or validation errors."""

    def __init__(self, errors: list[tuple[int, str]]) -> None:
        self.errors = errors
        formatted = "\n".join(f"Line {line_num}: {msg}" for line_num, msg in errors)
        super().__init__(
            f"Golden dataset validation failed with {len(errors)} error(s):\n{formatted}"
        )


def span_in_text(span: str, text: str) -> bool:
    """Return True if normalized span appears in normalized text (case/whitespace-insensitive)."""
    norm_span = _WS_RE.sub(" ", normalize_text(span)).strip().casefold()
    norm_text = _WS_RE.sub(" ", normalize_text(text)).strip().casefold()
    if not norm_span:
        return False
    return norm_span in norm_text


def load_golden(path: Path, *, reviewed_only: bool = False) -> list[GoldenItem]:
    """Parse a golden JSONL file, collecting ALL line-numbered errors before raising.

    Args:
        path: Path to the JSONL file.
        reviewed_only: When True, return only items with ``reviewed=True``.

    Returns:
        List of validated ``GoldenItem`` instances.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        GoldenFormatError: If any line has invalid JSON, schema errors, duplicate IDs,
            or duplicate normalized questions.
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    errors: list[tuple[int, str]] = []
    items: list[GoldenItem] = []
    seen_ids: dict[str, int] = {}
    seen_questions: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                raw_obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append((line_num, f"Invalid JSON: {exc.msg}"))
                continue

            if not isinstance(raw_obj, dict):
                errors.append((line_num, f"Expected JSON object, got {type(raw_obj).__name__}"))
                continue

            try:
                item = GoldenItem.model_validate(raw_obj)
            except ValidationError as exc:
                msg = "; ".join(err["msg"] for err in exc.errors())
                errors.append((line_num, f"Schema validation failed: {msg}"))
                continue

            if item.id in seen_ids:
                errors.append(
                    (line_num, f"Duplicate id '{item.id}' (first seen on line {seen_ids[item.id]})")
                )
            else:
                seen_ids[item.id] = line_num

            norm_q = normalize_text(item.question).casefold()
            # Allow authorized control items to share the exact question of their restricted item
            if norm_q in seen_questions and item.control_for is None:
                errors.append(
                    (
                        line_num,
                        f"Duplicate normalized question (first on line {seen_questions[norm_q]})",
                    )
                )
            elif item.control_for is None:
                seen_questions[norm_q] = line_num

            items.append(item)

    if errors:
        raise GoldenFormatError(errors)

    if reviewed_only:
        return [item for item in items if item.reviewed]
    return items


def _canonical_lines(items: list[GoldenItem]) -> list[str]:
    """Serialize items sorted by id with stable key ordering."""
    sorted_items = sorted(items, key=lambda item: item.id)
    return [
        json.dumps(item.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        for item in sorted_items
    ]


def write_golden(path: Path, items: list[GoldenItem]) -> None:
    """Write golden items in canonical sorted JSONL format with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _canonical_lines(items)
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")


def golden_file_hash(path: Path) -> str:
    """Compute SHA-256 hex digest of the canonical content of a golden JSONL file."""
    items = load_golden(path, reviewed_only=False)
    lines = _canonical_lines(items)
    canonical = "\n".join(lines) + ("\n" if lines else "")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
