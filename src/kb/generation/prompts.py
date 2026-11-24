"""Prompt loader and metadata parser enforcing Invariant I9 (Prompts are files)."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PromptMetadata(BaseModel):
    """Metadata extracted from prompt version header."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    purpose: str
    changelog: str = ""


class PromptLoadError(Exception):
    """Raised when a prompt file cannot be found or parsed."""


def get_prompts_dir() -> Path:
    """Return the absolute path to the prompts directory."""
    # Anchored to project root (2 levels up from src/kb/generation)
    return Path(__file__).resolve().parent.parent.parent.parent / "prompts"


def load_prompt(name: str, version: str) -> tuple[PromptMetadata, str]:
    """Load a versioned prompt file and parse its metadata header.

    Args:
        name: Base name of prompt (e.g. 'answer').
        version: Version string (e.g. '1.0.0').

    Returns:
        Tuple of (PromptMetadata, prompt_body_text).

    Raises:
        PromptLoadError: If file is missing or header is malformed.
    """
    prompts_dir = get_prompts_dir()
    prompt_file = prompts_dir / f"{name}.v{version}.md"

    if not prompt_file.exists():
        raise PromptLoadError(f"Prompt file not found: {prompt_file}")

    content = prompt_file.read_text(encoding="utf-8")

    # Parse frontmatter bounded by ---
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise PromptLoadError(
            f"Prompt file {prompt_file.name} missing frontmatter header opening '---'"
        )

    header_lines: list[str] = []
    body_lines: list[str] = []
    in_header = True

    for line in lines[1:]:
        if in_header:
            if line.strip() == "---":
                in_header = False
            else:
                header_lines.append(line)
        else:
            body_lines.append(line)

    if in_header:
        raise PromptLoadError(f"Prompt file {prompt_file.name} missing closing '---' for header")

    fields: dict[str, str] = {}
    for hline in header_lines:
        line_clean = hline.strip()
        if not line_clean or line_clean.startswith("#"):
            continue
        if ":" in line_clean:
            key, val = line_clean.split(":", 1)
            fields[key.strip()] = val.strip()

    metadata = PromptMetadata(
        name=fields.get("name", name),
        version=fields.get("version", version),
        purpose=fields.get("purpose", ""),
        changelog=fields.get("changelog", ""),
    )

    body_text = "\n".join(body_lines).strip()
    return metadata, body_text
