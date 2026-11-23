"""Unit tests for structure-aware deterministic chunker."""

from kb.domain.documents import Block, BlockType, ParsedDocument, TableData
from kb.ingestion.chunkers.recursive import ChunkingConfig, chunk
from kb.ingestion.tokens import ApproxTokenCounter


def test_chunker_determinism() -> None:
    """Verify that identical input and config produce identical ChunkDraft sequence."""
    doc = ParsedDocument(
        title="Technical Whitepaper",
        blocks=[
            Block(
                type=BlockType.HEADING,
                text="System Overview",
                level=1,
                page=1,
                section_path=["System Overview"],
            ),
            Block(
                type=BlockType.PARAGRAPH,
                text="The knowledge platform uses PostgreSQL with pgvector for unified storage.",
                page=1,
                section_path=["System Overview"],
            ),
            Block(
                type=BlockType.PARAGRAPH,
                text="Embeddings are generated locally using sentence-transformers.",
                page=2,
                section_path=["System Overview"],
            ),
        ],
    )

    config = ChunkingConfig(target_tokens=50, max_tokens=100, overlap_tokens=10, min_tokens=10)
    counter = ApproxTokenCounter()

    chunks_run1 = chunk(doc, config, counter)
    chunks_run2 = chunk(doc, config, counter)

    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2, strict=True):
        assert c1.ordinal == c2.ordinal
        assert c1.text == c2.text
        assert c1.token_count == c2.token_count
        assert c1.content_hash == c2.content_hash
        assert c1.section == c2.section


def test_chunker_never_splits_table_row_property() -> None:
    """Property test verifying that table rows are never split across chunks."""
    # Generate a table with 30 distinct identifiable rows
    header = ["ID", "Employee", "Department", "Salary"]
    rows = [
        [f"EMP_{i:03d}", f"Person_{i}", f"Dept_{i % 5}", f"${50000 + i * 1000}"]
        for i in range(1, 31)
    ]

    doc = ParsedDocument(
        title="Personnel",
        blocks=[
            Block(
                type=BlockType.TABLE,
                text="",
                page=1,
                section_path=["Personnel"],
                table=TableData(header=header, rows=rows),
            )
        ],
    )

    # Tight chunk token bounds to force multiple table chunks
    config = ChunkingConfig(target_tokens=60, max_tokens=80, overlap_tokens=0, min_tokens=10)
    counter = ApproxTokenCounter()
    chunks = chunk(doc, config, counter)

    assert len(chunks) > 1

    # Verify every serialized row appears in exactly ONE chunk in full
    for row in rows:
        expected_row_substr = (
            f"ID: {row[0]} | Employee: {row[1]} | Department: {row[2]} | Salary: {row[3]}"
        )
        matching_chunks = [c for c in chunks if expected_row_substr in c.text]
        assert len(matching_chunks) == 1, f"Row {expected_row_substr} was split or duplicated!"


def test_chunker_oversize_single_row_flagged() -> None:
    """Verify that a single row exceeding max_tokens is emitted with oversize metadata."""
    header = ["Key", "MassivePayload"]
    # Create row with text exceeding max_tokens (approx 150 tokens = ~600 chars)
    giant_text = "DataPayload " * 60
    rows = [
        ["Row1", "Normal text"],
        ["Row2", giant_text],
        ["Row3", "Normal text"],
    ]

    doc = ParsedDocument(
        title="Huge Table",
        blocks=[
            Block(
                type=BlockType.TABLE,
                text="",
                page=1,
                section_path=["Huge Table"],
                table=TableData(header=header, rows=rows),
            )
        ],
    )

    config = ChunkingConfig(target_tokens=40, max_tokens=60, overlap_tokens=0, min_tokens=5)
    counter = ApproxTokenCounter()
    chunks = chunk(doc, config, counter)

    oversize_chunks = [c for c in chunks if c.metadata.get("oversize") is True]
    assert len(oversize_chunks) == 1
    assert "Row2" in oversize_chunks[0].text
    assert giant_text.strip() in oversize_chunks[0].text


def test_chunker_merges_tiny_trailing_chunk() -> None:
    """Verify tiny trailing chunks below min_tokens merge into the previous chunk in section."""
    doc = ParsedDocument(
        title="Document",
        blocks=[
            Block(
                type=BlockType.PARAGRAPH,
                text="Paragraph one with sufficient length to form the first chunk of the section. "
                * 3,
                page=1,
                section_path=["Main"],
            ),
            Block(
                type=BlockType.PARAGRAPH,
                text="Tiny tail.",  # very few tokens (< min_tokens)
                page=1,
                section_path=["Main"],
            ),
        ],
    )

    # target_tokens=40, max_tokens=200, min_tokens=15
    config = ChunkingConfig(target_tokens=40, max_tokens=200, overlap_tokens=0, min_tokens=15)
    counter = ApproxTokenCounter()
    chunks = chunk(doc, config, counter)

    # Instead of two chunks where the second is tiny, it should merge into 1 chunk
    assert len(chunks) == 1
    assert "Tiny tail." in chunks[0].text


def test_chunker_never_merges_across_top_level_sections() -> None:
    """Verify that chunker strictly respects top-level section boundaries."""
    doc = ParsedDocument(
        title="Multi-Section",
        blocks=[
            Block(
                type=BlockType.HEADING,
                text="Section Alpha",
                level=1,
                section_path=["Section Alpha"],
            ),
            Block(
                type=BlockType.PARAGRAPH,
                text="Short content in Alpha.",
                section_path=["Section Alpha"],
            ),
            Block(
                type=BlockType.HEADING,
                text="Section Beta",
                level=1,
                section_path=["Section Beta"],
            ),
            Block(
                type=BlockType.PARAGRAPH,
                text="Short content in Beta.",
                section_path=["Section Beta"],
            ),
        ],
    )

    # Even with large target_tokens, sections Alpha and Beta must never be merged
    config = ChunkingConfig(target_tokens=500, max_tokens=600, overlap_tokens=0, min_tokens=10)
    counter = ApproxTokenCounter()
    chunks = chunk(doc, config, counter)

    assert len(chunks) == 2
    assert chunks[0].section == "Section Alpha"
    assert "Section Alpha" in chunks[0].text
    assert "Section Beta" not in chunks[0].text

    assert chunks[1].section == "Section Beta"
    assert "Section Beta" in chunks[1].text
    assert "Section Alpha" not in chunks[1].text


def test_chunker_overlap_only_within_same_section() -> None:
    """Verify that overlap applies only between consecutive text chunks within the same section."""
    sentences = [f"This is sentence number {i} for section Alpha." for i in range(1, 15)]
    doc = ParsedDocument(
        title="Overlap Doc",
        blocks=[
            Block(
                type=BlockType.PARAGRAPH,
                text=" ".join(sentences),
                section_path=["Section Alpha"],
            ),
            Block(
                type=BlockType.PARAGRAPH,
                text="First sentence for section Beta.",
                section_path=["Section Beta"],
            ),
        ],
    )

    config = ChunkingConfig(target_tokens=40, max_tokens=60, overlap_tokens=15, min_tokens=5)
    counter = ApproxTokenCounter()
    chunks = chunk(doc, config, counter)

    # Chunks belonging to Alpha
    alpha_chunks = [c for c in chunks if c.section == "Section Alpha"]
    assert len(alpha_chunks) >= 2

    # Check that chunk 2 of Alpha contains overlap from chunk 1
    chunk1_last_sentence = alpha_chunks[0].text.split(". ")[-1]
    assert chunk1_last_sentence in alpha_chunks[1].text

    # Check Beta chunk has NO overlap from Alpha
    beta_chunks = [c for c in chunks if c.section == "Section Beta"]
    assert len(beta_chunks) == 1
    assert "section Alpha" not in beta_chunks[0].text
