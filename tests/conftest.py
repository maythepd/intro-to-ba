"""Keep every pytest API lifespan isolated from a developer's production .env."""
import pytest


@pytest.fixture(autouse=True)
def isolated_monitoring_database(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "api-telemetry.sqlite3").as_posix())
    monkeypatch.setenv("MONITORING_ADMIN_TOKEN", "test-only-token")
    monkeypatch.setenv("ENABLE_MONITORING_DEMO", "false")
