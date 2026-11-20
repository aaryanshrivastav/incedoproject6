"""FastAPI gateway application."""

from fastapi import FastAPI

app = FastAPI(
    title="Enterprise GenAI Knowledge Assistant",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint returning system status."""
    return {"status": "ok"}
