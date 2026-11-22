import httpx
import pytest
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers.llm_controller import (
    LLMController,
    LocalInferenceProvider,
    ModelUnavailableError,
)


def import_server_app_without_env_file(monkeypatch):
    """Import the API without invoking its legacy .env loader during tests."""
    import controllers.llm_controller as llm_module

    monkeypatch.setattr(llm_module, "load_env_file", lambda env_path=".env": None)
    return importlib.import_module("server.app")


class FakeResponse:
    def __init__(self, payload=None, status_error=None):
        self.payload = payload or {}
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return self.payload


def local_env(monkeypatch, **overrides):
    values = {
        "AIVCO_LOCAL_ENABLED": "true",
        "AIVCO_LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
        "AIVCO_LOCAL_MODEL": "test-local-model",
        "AIVCO_LOCAL_TIMEOUT": "2.5",
    }
    values.update(overrides)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_is_available_true_when_models_endpoint_responds(monkeypatch):
    local_env(monkeypatch)
    seen = {}

    def fake_get(url, timeout):
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeResponse({"data": []})

    monkeypatch.setattr("controllers.llm_controller.httpx.get", fake_get)
    provider = LocalInferenceProvider()
    assert provider.is_available() is True
    assert seen["url"] == "http://127.0.0.1:8080/v1/models"
    assert seen["timeout"].connect == pytest.approx(0.3)


@pytest.mark.parametrize("failure", [httpx.ConnectError("offline"), httpx.ReadTimeout("slow")])
def test_is_available_false_when_endpoint_unreachable(monkeypatch, failure):
    local_env(monkeypatch)

    def fail_get(*args, **kwargs):
        raise failure

    monkeypatch.setattr("controllers.llm_controller.httpx.get", fail_get)
    assert LocalInferenceProvider().is_available() is False


def test_is_available_false_when_disabled(monkeypatch):
    local_env(monkeypatch, AIVCO_LOCAL_ENABLED="false")
    monkeypatch.setattr(
        "controllers.llm_controller.httpx.get",
        lambda *args, **kwargs: pytest.fail("disabled provider must not probe"),
    )
    provider = LocalInferenceProvider()
    assert provider.is_available() is False


def test_generate_parses_openai_response_and_attributes_model(monkeypatch):
    local_env(monkeypatch)
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, body=json, timeout=timeout)
        return FakeResponse({
            "model": "test-local-model",
            "choices": [{"message": {"content": "  Local answer  "}}],
        })

    monkeypatch.setattr("controllers.llm_controller.httpx.post", fake_post)
    controller = LLMController(env_path="/nonexistent/test.env")
    result = controller.route_and_generate(
        "System instructions", "Current question", force_provider="local_llama"
    )
    assert result["response"] == "Local answer"
    assert result["provider_used"] == "local_llama"
    assert result["model_used"] == "test-local-model"
    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert seen["body"]["model"] == "test-local-model"
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "Current question"},
    ]


@pytest.mark.parametrize("failure", [httpx.ConnectError("offline"), httpx.ReadTimeout("slow")])
def test_local_failure_falls_back_to_groq(monkeypatch, failure):
    local_env(monkeypatch)
    monkeypatch.setattr(
        "controllers.llm_controller.httpx.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )
    controller = LLMController(env_path="/nonexistent/test.env")
    monkeypatch.setattr(controller, "_generate_groq", lambda system, user: "Cloud answer")
    result = controller.route_and_generate(
        "System", "Question", force_provider="local_llama"
    )
    assert result["response"] == "Cloud answer"
    assert result["provider_used"] == "groq (fallback from local)"
    assert result["fallback_triggered"] is True


def test_local_generate_raises_model_unavailable_on_http_error(monkeypatch):
    local_env(monkeypatch)
    monkeypatch.setattr(
        "controllers.llm_controller.httpx.post",
        lambda *args, **kwargs: FakeResponse(status_error=httpx.HTTPStatusError(
            "503", request=httpx.Request("POST", "http://localhost"),
            response=httpx.Response(503),
        )),
    )
    provider = LocalInferenceProvider()
    with pytest.raises(ModelUnavailableError, match="Local inference endpoint unavailable"):
        provider.generate("System", "Question")


def test_local_routing_uses_compact_budget_before_generation(monkeypatch):
    local_env(monkeypatch)
    controller = LLMController(env_path="/nonexistent/test.env")
    captured = {}

    def generate(system_prompt, user_prompt, history=None):
        captured.update(system=system_prompt, user=user_prompt, history=history)
        return "Local answer"

    monkeypatch.setattr(controller.local_provider, "generate", generate)
    old_history = [
        {"role": "user", "content": "OLD TURN " + "old " * 4_000},
        {"role": "assistant", "content": "OLD RESPONSE " + "answer " * 4_000},
        {"role": "user", "content": "RECENT TURN"},
        {"role": "assistant", "content": "RECENT RESPONSE"},
    ]
    result = controller.route_and_generate(
        "Protected system prompt", "Current user query", force_provider="local_llama",
        history=old_history,
    )
    assert result["response"] == "Local answer"
    assert "OLD TURN" not in str(captured["history"])
    assert "RECENT TURN" in str(captured["history"])
    assert captured["system"] == "Protected system prompt"
    assert captured["user"] == "Current user query"


def test_local_provider_hot_swap_updates_model_and_endpoint(monkeypatch):
    local_env(monkeypatch)
    controller = LLMController(env_path="/nonexistent/test.env")
    controller.set_local_model("mlx-qwen-8b", "http://127.0.0.1:8080/v1")
    assert controller.local_provider.model == "mlx-qwen-8b"
    assert controller.local_provider.base_url == "http://127.0.0.1:8080/v1"

    controller.set_local_model("ollama-qwen-8b", "http://127.0.0.1:11434/v1")
    assert controller.local_provider.model == "ollama-qwen-8b"
    assert controller.local_provider.base_url == "http://127.0.0.1:11434/v1"


def test_provider_status_endpoint_returns_live_provider_status(monkeypatch):
    server = import_server_app_without_env_file(monkeypatch)
    monkeypatch.setattr(server.llm_controller.local_provider, "is_available", lambda: True)
    monkeypatch.setattr(server.universal_web_agent, "is_connected", lambda: True)
    from fastapi.testclient import TestClient

    response = TestClient(server.app).get("/api/providers/status")
    assert response.status_code == 200
    providers = response.json()["providers"]
    assert set(providers) == {"local", "groq", "gemini", "browser"}
    assert providers["local"]["available"] is True
    assert providers["local"]["active_model"] == server.llm_controller.local_provider.model
    assert providers["local"]["base_url"] == server.llm_controller.local_provider.base_url
    assert "last_error" in providers["local"]
    assert "latency_ms" in providers["local"]
    assert providers["browser"]["available"] is False


def test_local_model_endpoint_hot_swaps_and_checks_reachability(monkeypatch):
    server = import_server_app_without_env_file(monkeypatch)
    monkeypatch.setattr(server.llm_controller.local_provider, "is_available", lambda: True)
    from fastapi.testclient import TestClient

    old_model = server.llm_controller.local_provider.model
    old_url = server.llm_controller.local_provider.base_url
    try:
        response = TestClient(server.app).post(
            "/api/providers/local/model",
            json={"model": "mlx-community/Qwen3-8B", "base_url": "http://127.0.0.1:8080/v1"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["provider"]["available"] is True
        assert body["provider"]["active_model"] == "mlx-community/Qwen3-8B"
        assert body["provider"]["base_url"] == "http://127.0.0.1:8080/v1"
    finally:
        server.llm_controller.set_local_model(old_model, old_url)


def test_local_model_endpoint_reports_unreachable_after_applying_config(monkeypatch):
    server = import_server_app_without_env_file(monkeypatch)
    monkeypatch.setattr(server.llm_controller.local_provider, "is_available", lambda: False)
    from fastapi.testclient import TestClient

    old_model = server.llm_controller.local_provider.model
    old_url = server.llm_controller.local_provider.base_url
    try:
        response = TestClient(server.app).post(
            "/api/providers/local/model",
            json={"model": "offline-test-model", "base_url": "http://127.0.0.1:11434/v1"},
        )
        assert response.status_code == 503
        assert response.json()["detail"]["provider"]["available"] is False
        assert server.llm_controller.local_provider.model == "offline-test-model"
    finally:
        server.llm_controller.set_local_model(old_model, old_url)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model": "   "},
        {"model": "qwen", "base_url": "https://example.com/v1"},
        {"model": "qwen", "base_url": "http://127.0.0.1:9000/v1"},
    ],
)
def test_local_model_endpoint_rejects_empty_or_invalid_payloads(monkeypatch, payload):
    server = import_server_app_without_env_file(monkeypatch)
    from fastapi.testclient import TestClient

    response = TestClient(server.app).post("/api/providers/local/model", json=payload)
    assert response.status_code == 422
