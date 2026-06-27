from pathlib import Path

import pytest

import web.app as web_app


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def protected_api_state(monkeypatch):
    original_testing = web_app.app.config.get("TESTING", False)
    monkeypatch.setattr(web_app, "WEB_API_KEY", None)
    monkeypatch.setattr(
        web_app,
        "_get_gpu_stats",
        lambda: {"cuda_available": False, "gpus": [], "error": None},
    )
    web_app.app.config["TESTING"] = True
    yield
    web_app.app.config["TESTING"] = original_testing


@pytest.fixture
def client():
    return web_app.app.test_client()


def test_gpu_stats_allows_loopback_without_api_key(client):
    response = client.get(
        "/api/gpu-stats",
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "cuda_available": False,
        "gpus": [],
        "error": None,
    }


def test_gpu_stats_blocks_remote_without_api_key(client):
    response = client.get(
        "/api/gpu-stats",
        environ_overrides={"REMOTE_ADDR": "203.0.113.10"},
    )

    assert response.status_code == 403
    assert response.get_json()["error"] == "Set WEB_API_KEY to allow non-local API access."


def test_gpu_stats_requires_configured_api_key(client, monkeypatch):
    monkeypatch.setattr(web_app, "WEB_API_KEY", "test-secret")

    missing = client.get(
        "/api/gpu-stats",
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )
    bearer = client.get(
        "/api/gpu-stats",
        headers={"Authorization": "Bearer test-secret"},
        environ_overrides={"REMOTE_ADDR": "203.0.113.10"},
    )
    api_key = client.get(
        "/api/gpu-stats",
        headers={"X-API-Key": "test-secret"},
        environ_overrides={"REMOTE_ADDR": "203.0.113.10"},
    )

    assert missing.status_code == 401
    assert bearer.status_code == 200
    assert api_key.status_code == 200


def test_ui_provides_in_memory_api_key_field_and_auth_headers(client):
    html = client.get("/").get_data(as_text=True)
    app_js = (ROOT / "web/static/app.js").read_text()

    assert 'id="api-key-input"' in html
    assert 'type="password"' in html
    assert 'autocomplete="off"' in html
    assert "localStorage" not in app_js
    assert "sessionStorage" not in app_js
    assert 'fetch("/api/gpu-stats", { headers: apiHeaders() })' in app_js
    assert 'headers: apiHeaders({ "Content-Type": "application/json" })' in app_js
    assert 'headers.Authorization = "Bearer " + apiKey' in app_js
