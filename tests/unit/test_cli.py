"""Unit tests for the CLI entry points."""

from typer.testing import CliRunner

from kb import __version__
from kb.cli import app

runner = CliRunner()


def test_kb_version_command() -> None:
    """The kb version command prints the correct version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"kb version {__version__}" in result.stdout
