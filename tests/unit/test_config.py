"""Unit tests for configuration loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kb.config import Settings


def test_valid_environment_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings loads successfully when all required environment variables are set."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://kb:kb@localhost:5432/kb")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-12345")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)
    assert settings.DATABASE_URL == "postgresql+psycopg://kb:kb@localhost:5432/kb"
    assert settings.REDIS_URL == "redis://localhost:6379/0"
    assert settings.JWT_SECRET == "test-secret-key-12345"
    assert settings.EMBEDDING_DIM == 1536
    assert settings.APP_ENV == "test"
    assert settings.LOG_LEVEL == "DEBUG"


@pytest.mark.parametrize(
    "missing_var",
    ["DATABASE_URL", "REDIS_URL", "JWT_SECRET", "EMBEDDING_DIM"],
)
def test_missing_required_variable_fails_fast(
    missing_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing any required variable raises ValidationError naming the missing variable."""
    env_vars = {
        "DATABASE_URL": "postgresql+psycopg://kb:kb@localhost:5432/kb",
        "REDIS_URL": "redis://localhost:6379/0",
        "JWT_SECRET": "secret",
        "EMBEDDING_DIM": "1536",
    }
    del env_vars[missing_var]

    # Clear environment of required variables
    for key in ["DATABASE_URL", "REDIS_URL", "JWT_SECRET", "EMBEDDING_DIM"]:
        monkeypatch.delenv(key, raising=False)

    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    error_text = str(exc_info.value)
    assert missing_var in error_text, f"Error message should mention {missing_var}: {error_text}"


def test_invalid_embedding_dim_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """EMBEDDING_DIM <= 0 raises ValidationError."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://kb:kb@localhost:5432/kb")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("JWT_SECRET", "secret")
    monkeypatch.setenv("EMBEDDING_DIM", "0")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "EMBEDDING_DIM" in str(exc_info.value)


def test_env_example_covers_all_config_fields() -> None:
    """.env.example must cover every variable defined in Settings."""
    env_example_path = Path(__file__).parents[2] / ".env.example"
    assert env_example_path.exists(), ".env.example must exist"

    example_vars: set[str] = set()
    for line in env_example_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            var_name = line.split("=", 1)[0].strip()
            example_vars.add(var_name)

    settings_fields = set(Settings.model_fields.keys())

    missing_in_example = settings_fields - example_vars
    assert not missing_in_example, (
        f".env.example is missing variables defined in Settings: {missing_in_example}"
    )
