import pytest
from app import __version__
from app.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient


def test_health_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "service": "stratosense-api",
        "version": __version__,
        "environment": "test",
    }


def test_ready_is_503_and_honest_when_checkpoint_missing(client):
    r = client.get("/api/v1/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False and body["model_loaded"] is False
    assert "No model checkpoint found" in body["reason"]


def test_ready_never_claims_ready_just_because_a_file_exists(tmp_path):
    ckpt = tmp_path / "fake.pth"
    ckpt.write_bytes(b"not a real model")
    app = create_app(Settings(app_env="test", model_path=ckpt, _env_file=None))
    r = TestClient(app).get("/api/v1/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False and body["model_loaded"] is False
    assert "not implemented" in body["reason"]


def test_cors_allows_configured_origin_only(client):
    ok = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.get("/api/v1/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_unknown_route_is_404(client):
    assert client.get("/api/v1/nope").status_code == 404


def test_unhandled_errors_do_not_leak_tracebacks(settings):
    app = create_app(settings)

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret internal detail")

    r = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error."}
    assert "secret" not in r.text


def test_cors_origin_parsing_strips_whitespace_and_trailing_slash():
    s = Settings(cors_origins=" https://a.app/ , http://localhost:5173 ", _env_file=None)
    assert s.cors_origin_list == ["https://a.app", "http://localhost:5173"]


def test_production_rejects_wildcard_or_empty_cors():
    with pytest.raises(ValueError):
        Settings(app_env="production", cors_origins="*", _env_file=None)
    with pytest.raises(ValueError):
        Settings(app_env="production", cors_origins="", _env_file=None)
    assert Settings(app_env="production", cors_origins="https://x.app", _env_file=None)


def test_upload_limit_must_be_positive():
    with pytest.raises(ValueError):
        Settings(max_upload_mb=0, _env_file=None)
    assert Settings(max_upload_mb=7, _env_file=None).max_upload_bytes == 7 * 1024 * 1024


def test_relative_model_path_resolves_against_repo_root():
    s = Settings(_env_file=None)
    assert s.resolved_model_path.is_absolute()
    assert s.resolved_model_path.parts[-3:] == ("artifacts", "checkpoints", "stratosense_unet.pth")
