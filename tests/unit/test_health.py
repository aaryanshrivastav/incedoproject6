"""Unit tests for the FastAPI health check endpoint."""

from kb.api.app import app, health


def test_health_endpoint_direct() -> None:
    """The health handler returns status ok."""
    result = health()
    assert result == {"status": "ok"}


def test_health_route_in_fastapi_app() -> None:
    """The /health route is registered on the FastAPI app with GET method."""
    route = next(
        (
            r
            for r in app.routes
            if getattr(r, "path", None) == "/health" and "GET" in getattr(r, "methods", set())
        ),
        None,
    )
    assert route is not None, "/health GET route must be registered in app"
