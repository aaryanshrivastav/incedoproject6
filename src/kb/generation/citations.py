"""Citation parsing, validation, and stored snippet resolution."""

import json
import re
from typing import Any
from uuid import UUID

from kb.domain.answers import Citation
from kb.domain.retrieval import ContextBundle

CHUNK_MARKER_PATTERN = re.compile(r"\[\[chunk:([a-zA-Z0-9_\-]+)\]\]")


class ModelOutputParseError(Exception):
    """Raised when the model output cannot be parsed into the expected JSON schema."""


def parse_model_output(raw_text: str) -> dict[str, Any]:
    """Parse JSON object from model response, stripping markdown code fences if present.

    Args:
        raw_text: Raw output string from LLM.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ModelOutputParseError: If text is not valid JSON or not a dictionary.
    """
    text = raw_text.strip()

    # Strip markdown code fence if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening ``` or ```json
        if len(lines) > 1:
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            text = "\n".join(lines).strip()

    # Find outermost JSON object if surrounded by preamble/postamble
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ModelOutputParseError(f"Model output is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ModelOutputParseError("Model output JSON must be an object (dict)")

    return data


def extract_and_validate_citations(
    answer: str,
    declared_citations: list[str],
    valid_chunk_ids: set[UUID],
) -> tuple[str, list[UUID], list[str]]:
    """Extract and validate citation markers, stripping any invalid or unrecognized IDs.

    Args:
        answer: Raw answer string containing potential [[chunk:<id>]] markers.
        declared_citations: Chunk IDs declared in model's JSON "citations" field.
        valid_chunk_ids: Set of chunk UUIDs present in the prompt's ContextBundle.

    Returns:
        Tuple of:
          - sanitized_answer: answer text with invalid [[chunk:<id>]] markers removed.
          - valid_ids: list of unique valid chunk UUIDs cited.
          - invalid_citations: list of invalid or unrecognized ID strings.
    """
    found_markers = CHUNK_MARKER_PATTERN.findall(answer)
    all_raw_ids = list(declared_citations) + found_markers

    valid_ids: list[UUID] = []
    seen_valid: set[UUID] = set()
    invalid_ids: list[str] = []
    seen_invalid: set[str] = set()

    for raw_id in all_raw_ids:
        raw_clean = str(raw_id).strip()
        try:
            parsed_uuid = UUID(raw_clean)
            if parsed_uuid in valid_chunk_ids:
                if parsed_uuid not in seen_valid:
                    seen_valid.add(parsed_uuid)
                    valid_ids.append(parsed_uuid)
            else:
                if raw_clean not in seen_invalid:
                    seen_invalid.add(raw_clean)
                    invalid_ids.append(raw_clean)
        except (ValueError, TypeError):
            if raw_clean not in seen_invalid:
                seen_invalid.add(raw_clean)
                invalid_ids.append(raw_clean)

    # Sanitize answer text: strip unknown/invalid markers
    def _replace_marker(match: re.Match[str]) -> str:
        marker_id = match.group(1).strip()
        try:
            if UUID(marker_id) in valid_chunk_ids:
                return match.group(0)  # Keep valid marker
        except (ValueError, TypeError):
            pass
        return ""  # Strip invalid marker

    sanitized_answer = CHUNK_MARKER_PATTERN.sub(_replace_marker, answer)
    # Clean up any leftover double spaces from stripped markers
    sanitized_answer = re.sub(r" +", " ", sanitized_answer).strip()

    return sanitized_answer, valid_ids, invalid_ids


def make_snippet(text: str, max_chars: int = 240) -> tuple[str, bool]:
    """Create a snippet cut at a word boundary strictly as an exact prefix of text.

    Args:
        text: Full chunk text.
        max_chars: Maximum character limit.

    Returns:
        Tuple of (snippet, is_truncated).
    """
    if len(text) <= max_chars:
        return text, False

    # Cut up to max_chars
    candidate = text[:max_chars]
    # Find last word boundary
    last_space = candidate.rfind(" ")
    if last_space > 0:
        snippet = candidate[:last_space]
    else:
        snippet = candidate

    return snippet, True


def resolve_citations(
    valid_ids: list[UUID],
    bundle: ContextBundle,
    max_chars: int = 240,
) -> list[Citation]:
    """Resolve validated chunk UUIDs to Citation objects with stored chunk text provenance.

    Args:
        valid_ids: List of validated chunk UUIDs.
        bundle: ContextBundle containing stored chunk metadata and text.
        max_chars: Snippet character limit.

    Returns:
        List of Citation models.
    """
    item_map = {item.chunk_id: item for item in bundle.items}
    citations: list[Citation] = []

    for cid in valid_ids:
        if cid in item_map:
            item = item_map[cid]
            snippet, truncated = make_snippet(item.text, max_chars=max_chars)
            citations.append(
                Citation(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    document_title=item.document_title,
                    source_uri=item.source_uri,
                    page=item.page,
                    section=item.section,
                    snippet=snippet,
                    snippet_truncated=truncated,
                )
            )

    return citations
