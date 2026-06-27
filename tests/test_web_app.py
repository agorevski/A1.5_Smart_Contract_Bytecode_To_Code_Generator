import json
import shutil
from pathlib import Path
from types import SimpleNamespace

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


def test_health_surfaces_readiness_config_and_limits(client, monkeypatch):
    monkeypatch.setattr(web_app, "decompiler", None)
    monkeypatch.setattr(web_app, "active_model_path", "models/missing")
    monkeypatch.setattr(web_app, "model_load_error", "not found")
    monkeypatch.setattr(web_app, "tac_lookup", None)

    data = client.get("/api/health").get_json()

    assert data["status"] == "ok"
    assert data["ready"] is False
    assert data["model_path"] == "models/missing"
    assert data["model_error"] == "not found"
    assert data["limits"]["max_bytecode_hex_length"] == web_app.MAX_BYTECODE_HEX_LENGTH
    assert data["generation_defaults"]["max_new_tokens"] >= 1


def test_ui_exposes_readiness_generation_cancel_and_security_controls(client):
    html = client.get("/").get_data(as_text=True)
    app_js = (ROOT / "web/static/app.js").read_text()

    assert 'id="readiness-panel"' in html
    assert 'id="bytecode-guidance"' in html
    assert 'id="max-new-tokens-input"' in html
    assert 'id="btn-cancel-decompile"' in html
    assert 'id="btn-vulnerability-scan"' in html
    assert 'id="btn-classify"' in html
    assert 'id="btn-audit-report"' in html
    assert 'fetch("/api/health", { headers: apiHeaders() })' in app_js
    assert 'runSecurityEndpoint("vulnerability scan", "/api/vulnerability-scan")' in app_js


def _parse_sse_events(text):
    events = []
    for raw in text.strip().split("\n\n"):
        event_type = None
        data_lines = []
        for line in raw.splitlines():
            if line.startswith("event: "):
                event_type = line.split(": ", 1)[1]
            elif line.startswith("data: "):
                data_lines.append(line.split(": ", 1)[1])
        if event_type and data_lines:
            events.append((event_type, json.loads("\n".join(data_lines))))
    return events


def test_decompile_accepts_generation_controls_and_writes_trace(client, monkeypatch):
    trace_dir = ROOT / "results" / "test_inference_traces"
    shutil.rmtree(trace_dir, ignore_errors=True)

    class FakeAnalyzer:
        def __init__(self, bytecode):
            self.instructions = [object(), object()]
            self.basic_blocks = {"b0": object()}
            self.functions = {
                "func_abcdef01": SimpleNamespace(
                    name="func_abcdef01",
                    visibility="public",
                    is_payable=False,
                    is_view=True,
                )
            }

        def generate_per_function_tac(self):
            return {"func_abcdef01": "func_abcdef01:\n  v0 = CALLVALUE\n  RETURN v0"}

    class FakeDecompiler:
        def __init__(self):
            self.calls = []

        def _count_tokens(self, text):
            return len(text.split())

        def _tac_token_budget(self, metadata=None, max_new_tokens=1024):
            return 8

        def _truncate_tac(self, tac_text, max_tokens):
            return "\n".join(tac_text.splitlines()[:2])

        def _build_prompt(self, tac_text, metadata=None, max_new_tokens=1024):
            return "PROMPT " + self._truncate_tac(tac_text, 8)

        def decompile_tac_to_solidity(self, tac, metadata=None, **kwargs):
            self.calls.append({"tac": tac, "metadata": metadata, "kwargs": kwargs})
            return "function func_abcdef01() public view returns (uint256) { return 0; }"

        def _assemble_contract(self, functions, analyzer):
            return "contract DecompiledContract {\\n" + "\\n".join(functions.values()) + "\\n}"

    class FakeResolver:
        def resolve_function_names(self, fnames):
            return {
                fname: SimpleNamespace(
                    to_dict=lambda fname=fname: {
                        "best_match": {
                            "selector": "0xabcdef01",
                            "signature": "func()",
                            "confidence": 95,
                            "source": "builtin",
                        },
                        "candidates": [],
                    }
                )
                for fname in fnames
            }

    fake_decompiler = FakeDecompiler()
    monkeypatch.setattr(web_app, "BytecodeAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(web_app, "get_resolver", lambda use_remote=False: FakeResolver())
    monkeypatch.setattr(web_app, "decompiler", fake_decompiler)
    monkeypatch.setattr(web_app, "model_config_dict", {"model_name": "fake"})
    monkeypatch.setattr(web_app, "active_model_path", "models/fake")
    monkeypatch.setattr(web_app, "model_load_error", None)
    monkeypatch.setattr(web_app, "tac_lookup", None)
    monkeypatch.setattr(web_app, "INFERENCE_TRACE_ENABLED", True)
    monkeypatch.setattr(web_app, "INFERENCE_TRACE_DIR", str(trace_dir))

    response = client.post(
        "/api/decompile",
        json={
            "bytecode": "0x6000",
            "generation": {
                "max_new_tokens": 64,
                "temperature": 0.2,
                "do_sample": True,
                "repetition_penalty": 1.05,
            },
        },
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.get_data(as_text=True))
    result = [data for event, data in events if event == "result"][-1]

    assert result["request_id"]
    assert result["success"] is True
    assert result["effective_generation_config"]["max_new_tokens"] == 64
    assert fake_decompiler.calls[0]["kwargs"]["max_new_tokens"] == 64
    assert fake_decompiler.calls[0]["kwargs"]["temperature"] == 0.2
    assert result["function_results"][0]["source"] == "model_inference"
    assert result["function_results"][0]["diagnostics"]["tac_truncated"] is True
    trace_path = ROOT / result["trace_path"]
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text())
    assert trace["request_id"] == result["request_id"]
    assert trace["functions"]["func_abcdef01"]["diagnostics"]["generated_tokens"] > 0

    shutil.rmtree(trace_dir, ignore_errors=True)


def test_decompile_rejects_invalid_generation_controls(client):
    response = client.post(
        "/api/decompile",
        json={"bytecode": "0x6000", "generation": {"max_new_tokens": 0}},
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )

    assert response.status_code == 400
    assert "max_new_tokens" in response.get_json()["error"]


def test_decompile_oversized_bytecode_returns_413(client, monkeypatch):
    monkeypatch.setattr(web_app, "MAX_BYTECODE_HEX_LENGTH", 4)

    response = client.post(
        "/api/decompile",
        json={"bytecode": "0x600000"},
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )

    assert response.status_code == 413
    assert "too large" in response.get_json()["error"]
