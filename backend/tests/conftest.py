import pytest
from app.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        app_env="test",
        model_path=tmp_path / "missing" / "model.pth",
        cors_origins="http://localhost:5173",
        _env_file=None,
    )


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings), raise_server_exceptions=False)
