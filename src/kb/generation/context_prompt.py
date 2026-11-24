"""Renderer for assembling ContextBundle into an untrusted XML-like context block.

Enforces Invariant I3: Untrusted document text is safely delimited inside <context>
and all closing tag sequences are neutralized to prevent delimiter breakout.
"""

from kb.domain.retrieval import ContextBundle


def escape_chunk_text(text: str) -> str:
    """Neutralize potential closing tag sequences inside chunk text."""
    return (
        text.replace("</context>", "&lt;/context&gt;")
        .replace("</chunk>", "&lt;/chunk&gt;")
        .replace("<context>", "&lt;context&gt;")
    )


def render_context(bundle: ContextBundle) -> str:
    """Render ContextBundle items into a structured <context> XML block.

    Args:
        bundle: Assembled context bundle containing ordered, budgeted chunks.

    Returns:
        String containing the formatted <context> block, or empty string if no items.
    """
    if not bundle.items:
        return "<context>\n</context>"

    lines: list[str] = ["<context>"]
    for item in bundle.items:
        attrs = [f'id="{item.chunk_id}"', f'title="{item.document_title}"']
        if item.page is not None:
            attrs.append(f'page="{item.page}"')
        if item.section:
            attrs.append(f'section="{item.section}"')

        attr_str = " ".join(attrs)
        escaped_body = escape_chunk_text(item.text)
        lines.append(f"<chunk {attr_str}>")
        lines.append(escaped_body)
        lines.append("</chunk>")

    lines.append("</context>")
    return "\n".join(lines)
