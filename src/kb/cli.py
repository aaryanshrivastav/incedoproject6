"""Command line interface for the knowledge assistant."""

import typer

from kb import __version__

app = typer.Typer(
    name="kb",
    help="Enterprise GenAI Knowledge Assistant CLI",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Enterprise GenAI Knowledge Assistant."""


@app.command()
def version() -> None:
    """Print the application version."""
    typer.echo(f"kb version {__version__}")


if __name__ == "__main__":
    app()
