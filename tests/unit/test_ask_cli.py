"""Unit tests for the 'kb ask' command line interface."""

from typer.testing import CliRunner

from kb.cli import app

runner = CliRunner()


def test_ask_cli_help() -> None:
    """Verify kb ask --help displays options and trusted dev tool disclaimer."""
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--hub" in result.stdout
    assert "--user-tag" in result.stdout
    assert "--show-context" in result.stdout
    assert "--json" in result.stdout
    assert "trusted development tool" in result.stdout


def test_ask_cli_missing_question_fails() -> None:
    """Verify invoking ask without question argument fails with usage error."""
    result = runner.invoke(app, ["ask"])
    assert result.exit_code != 0
    output = result.output or ""
    assert "Missing argument" in output or "Usage:" in output or result.exit_code == 2
