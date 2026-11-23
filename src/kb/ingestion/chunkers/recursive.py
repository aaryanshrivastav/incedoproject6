"""Structure-aware deterministic document chunker."""

import hashlib
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kb.domain.documents import Block, BlockType, ParsedDocument
from kb.ingestion.normalize import normalize_text
from kb.ingestion.tokens import ApproxTokenCounter, TokenCounter

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_SPLIT_RE = re.compile(r"\s+")


class ChunkingConfig(BaseModel):
    """Configuration parameters for document chunking."""

    model_config = ConfigDict(frozen=True)

    target_tokens: int = Field(default=400, ge=1)
    max_tokens: int = Field(default=512, ge=1)
    overlap_tokens: int = Field(default=50, ge=0)
    min_tokens: int = Field(default=40, ge=0)


class ChunkDraft(BaseModel):
    """Draft chunk entity produced by the chunker before embedding and database upsert."""

    model_config = ConfigDict(frozen=True)

    ordinal: int = Field(ge=0, description="0-based sequence ordinal")
    text: str = Field(description="Normalized chunk text")
    page: int | None = Field(default=None, description="Starting 1-based page number")
    page_end: int | None = Field(default=None, description="Ending 1-based page number")
    section: str | None = Field(default=None, description="Joined section path hierarchy")
    token_count: int = Field(ge=0, description="Token count of chunk text")
    content_hash: str = Field(description="SHA-256 hash of chunk text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


def _split_oversize_text(text: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """Split oversize text into pieces strictly within max_tokens."""
    if counter.count_tokens(text) <= max_tokens:
        return [text]

    # 1. Try splitting into sentences
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text]

    pieces: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = counter.count_tokens(sentence)
        if s_tokens > max_tokens:
            # Flush current accumulation if any
            if current_sentences:
                pieces.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0

            # 2. Split oversize sentence by words
            words = _WORD_SPLIT_RE.split(sentence)
            curr_words: list[str] = []
            curr_w_tokens = 0
            for w in words:
                w_tok = counter.count_tokens(w + " ")
                if curr_w_tokens + w_tok > max_tokens and curr_words:
                    pieces.append(" ".join(curr_words))
                    curr_words = [w]
                    curr_w_tokens = w_tok
                else:
                    curr_words.append(w)
                    curr_w_tokens += w_tok
            if curr_words:
                pieces.append(" ".join(curr_words))
        else:
            if current_tokens + s_tokens > max_tokens and current_sentences:
                pieces.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_tokens = s_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += s_tokens

    if current_sentences:
        pieces.append(" ".join(current_sentences))

    return pieces


def _get_overlap_prefix(text: str, overlap_tokens: int, counter: TokenCounter) -> str:
    """Extract up to overlap_tokens from the end of text to prepend to the next chunk."""
    if overlap_tokens <= 0 or not text:
        return ""

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return ""

    # Accumulate trailing sentences up to overlap_tokens
    overlap_sentences: list[str] = []
    accum_tokens = 0
    for s in reversed(sentences):
        s_tok = counter.count_tokens(s)
        if accum_tokens + s_tok > overlap_tokens and overlap_sentences:
            break
        overlap_sentences.insert(0, s)
        accum_tokens += s_tok

    return " ".join(overlap_sentences)


def chunk(
    parsed: ParsedDocument,
    config: ChunkingConfig | None = None,
    counter: TokenCounter | None = None,
) -> list[ChunkDraft]:
    """Chunk a ParsedDocument into deterministic ChunkDraft items according to structure.

    Rules:
    - Never merge across top-level sections.
    - Pack paragraphs up to target_tokens. Split on sentences/words if exceeding max_tokens.
    - Overlap applies only between consecutive text chunks within the same section.
    - Tables: rows formatted as "Header: val | Header: val", packed up to max_tokens.
      Never split a table row. Oversize single rows get their own chunk with metadata
      {"oversize": True}.
    - Merge trailing chunks below min_tokens into previous chunk if it fits.
    """
    if config is None:
        config = ChunkingConfig()
    if counter is None:
        counter = ApproxTokenCounter()

    if not parsed.blocks:
        return []

    # 1. Group blocks by top-level section
    # A top-level section is block.section_path[0] if available, otherwise ""
    section_groups: list[list[Block]] = []
    current_group: list[Block] = []
    current_top_section: str | None = None

    for block in parsed.blocks:
        top_sec = block.section_path[0] if block.section_path else ""
        if current_top_section is None or top_sec == current_top_section:
            current_group.append(block)
            current_top_section = top_sec
        else:
            if current_group:
                section_groups.append(current_group)
            current_group = [block]
            current_top_section = top_sec

    if current_group:
        section_groups.append(current_group)

    drafts: list[ChunkDraft] = []
    current_ordinal = 0

    # 2. Process each section group independently
    for group in section_groups:
        group_chunks: list[dict[str, Any]] = []

        idx = 0
        while idx < len(group):
            block = group[idx]

            if block.type == BlockType.TABLE and block.table is not None:
                # Handle Table Block
                headers = block.table.header
                rows = block.table.rows

                if not rows:
                    idx += 1
                    continue

                # Serialize each row individually
                serialized_rows: list[str] = []
                for row in rows:
                    parts: list[str] = []
                    for c_idx, cell in enumerate(row):
                        h = (
                            headers[c_idx]
                            if c_idx < len(headers) and headers[c_idx]
                            else f"Col{c_idx + 1}"
                        )
                        parts.append(f"{h}: {cell}")
                    if parts:
                        serialized_rows.append(" | ".join(parts))

                curr_table_rows: list[str] = []
                curr_table_tokens = 0

                for s_row in serialized_rows:
                    row_tokens = counter.count_tokens(s_row)

                    # Check for oversize single row
                    if row_tokens > config.max_tokens:
                        # Flush current accumulated rows first
                        if curr_table_rows:
                            tbl_text = "\n".join(curr_table_rows)
                            group_chunks.append(
                                {
                                    "text": tbl_text,
                                    "page": block.page,
                                    "page_end": block.page,
                                    "section": " > ".join(block.section_path)
                                    if block.section_path
                                    else None,
                                    "metadata": {"table_name": block.table.name},
                                }
                            )
                            curr_table_rows = []
                            curr_table_tokens = 0

                        # Emit oversize single row directly
                        group_chunks.append(
                            {
                                "text": s_row,
                                "page": block.page,
                                "page_end": block.page,
                                "section": " > ".join(block.section_path)
                                if block.section_path
                                else None,
                                "metadata": {"oversize": True, "table_name": block.table.name},
                            }
                        )
                        continue

                    # Normal row packing
                    if curr_table_tokens + row_tokens > config.max_tokens and curr_table_rows:
                        tbl_text = "\n".join(curr_table_rows)
                        group_chunks.append(
                            {
                                "text": tbl_text,
                                "page": block.page,
                                "page_end": block.page,
                                "section": " > ".join(block.section_path)
                                if block.section_path
                                else None,
                                "metadata": {"table_name": block.table.name},
                            }
                        )
                        curr_table_rows = [s_row]
                        curr_table_tokens = row_tokens
                    else:
                        curr_table_rows.append(s_row)
                        curr_table_tokens += row_tokens

                if curr_table_rows:
                    tbl_text = "\n".join(curr_table_rows)
                    group_chunks.append(
                        {
                            "text": tbl_text,
                            "page": block.page,
                            "page_end": block.page,
                            "section": " > ".join(block.section_path)
                            if block.section_path
                            else None,
                            "metadata": {"table_name": block.table.name},
                        }
                    )

                idx += 1

            else:
                # Handle Text Blocks (Heading, Paragraph, List Item)
                curr_text_pieces: list[str] = []
                curr_text_tokens = 0
                chunk_page_start: int | None = block.page
                chunk_page_end: int | None = block.page
                chunk_section = " > ".join(block.section_path) if block.section_path else None

                while idx < len(group) and group[idx].type != BlockType.TABLE:
                    b = group[idx]
                    b_text = normalize_text(b.text)
                    if not b_text:
                        idx += 1
                        continue

                    if chunk_page_start is None:
                        chunk_page_start = b.page
                    if b.page is not None:
                        chunk_page_end = b.page
                    if b.section_path:
                        chunk_section = " > ".join(b.section_path)

                    b_tokens = counter.count_tokens(b_text)

                    if b_tokens > config.max_tokens:
                        # Split oversize block
                        pieces = _split_oversize_text(b_text, config.max_tokens, counter)
                        for piece in pieces:
                            p_tok = counter.count_tokens(piece)
                            if curr_text_tokens + p_tok > config.target_tokens and curr_text_pieces:
                                # Finalize current text chunk
                                full_chunk_text = "\n\n".join(curr_text_pieces)
                                group_chunks.append(
                                    {
                                        "text": full_chunk_text,
                                        "page": chunk_page_start,
                                        "page_end": chunk_page_end,
                                        "section": chunk_section,
                                        "metadata": {},
                                        "is_text": True,
                                    }
                                )

                                # Prepare overlap prefix
                                overlap_prefix = _get_overlap_prefix(
                                    full_chunk_text, config.overlap_tokens, counter
                                )
                                curr_text_pieces = (
                                    [overlap_prefix, piece] if overlap_prefix else [piece]
                                )
                                curr_text_tokens = counter.count_tokens(
                                    "\n\n".join(curr_text_pieces)
                                )
                                chunk_page_start = b.page
                            else:
                                curr_text_pieces.append(piece)
                                curr_text_tokens += p_tok
                    else:
                        if curr_text_tokens + b_tokens > config.target_tokens and curr_text_pieces:
                            # Finalize current text chunk
                            full_chunk_text = "\n\n".join(curr_text_pieces)
                            group_chunks.append(
                                {
                                    "text": full_chunk_text,
                                    "page": chunk_page_start,
                                    "page_end": chunk_page_end,
                                    "section": chunk_section,
                                    "metadata": {},
                                    "is_text": True,
                                }
                            )

                            # Prepare overlap prefix for consecutive text chunks
                            overlap_prefix = _get_overlap_prefix(
                                full_chunk_text, config.overlap_tokens, counter
                            )
                            curr_text_pieces = (
                                [overlap_prefix, b_text] if overlap_prefix else [b_text]
                            )
                            curr_text_tokens = counter.count_tokens("\n\n".join(curr_text_pieces))
                            chunk_page_start = b.page
                        else:
                            curr_text_pieces.append(b_text)
                            curr_text_tokens += b_tokens

                    idx += 1

                if curr_text_pieces:
                    full_chunk_text = "\n\n".join(curr_text_pieces)
                    group_chunks.append(
                        {
                            "text": full_chunk_text,
                            "page": chunk_page_start,
                            "page_end": chunk_page_end,
                            "section": chunk_section,
                            "metadata": {},
                            "is_text": True,
                        }
                    )

        # 3. Trailing chunk merge logic within this section group
        if len(group_chunks) > 1:
            last_chunk = group_chunks[-1]
            last_tokens = counter.count_tokens(last_chunk["text"])

            if (
                last_tokens < config.min_tokens
                and not last_chunk["metadata"].get("oversize", False)
                and not group_chunks[-2]["metadata"].get("oversize", False)
            ):
                prev_chunk = group_chunks[-2]
                combined_text = prev_chunk["text"] + "\n\n" + last_chunk["text"]
                if counter.count_tokens(combined_text) <= config.max_tokens:
                    prev_chunk["text"] = combined_text
                    if last_chunk["page_end"] is not None:
                        prev_chunk["page_end"] = last_chunk["page_end"]
                    group_chunks.pop()

        # 4. Convert to ChunkDraft
        for c in group_chunks:
            normalized_c_text = normalize_text(c["text"])
            if not normalized_c_text:
                continue

            tok_count = counter.count_tokens(normalized_c_text)
            c_hash = hashlib.sha256(normalized_c_text.encode("utf-8")).hexdigest()

            drafts.append(
                ChunkDraft(
                    ordinal=current_ordinal,
                    text=normalized_c_text,
                    page=c["page"],
                    page_end=c["page_end"],
                    section=c["section"],
                    token_count=tok_count,
                    content_hash=c_hash,
                    metadata=c.get("metadata", {}),
                )
            )
            current_ordinal += 1

    return drafts
