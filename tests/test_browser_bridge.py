import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_websocket_registration_appears_in_provider_status(monkeypatch):
    import controllers.llm_controller as llm_module
    monkeypatch.setattr(llm_module, "load_env_file", lambda env_path=".env": None)
    server = importlib.import_module("server.app")
    from fastapi.testclient import TestClient

    client = TestClient(server.app)
    with client.websocket_connect("/ws/bridge") as websocket:
        websocket.send_json({
            "type": "register", "client": "chrome_extension", "version": "1.0.0",
            "tabs": {"claude": True, "chatgpt": False, "gemini": True, "grok": False},
        })
        status = client.get("/api/providers/status").json()["providers"]["browser"]
        assert status["available"] is True
        assert status["active_brains"] == ["claude", "gemini"]
        websocket.send_json({"type": "tabs_status", "tabs": {"claude": False, "gemini": False}})
        status2 = client.get("/api/providers/status").json()["providers"]["browser"]
        assert status2["available"] is False


def test_manifest_has_loopback_permissions_and_supported_brains():
    import json

    manifest_path = Path(__file__).resolve().parents[1] / "extension" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "http://127.0.0.1:8000/*" in manifest["host_permissions"]
    assert "http://localhost:8000/*" in manifest["host_permissions"]
    content_matches = " ".join(manifest["content_scripts"][0]["matches"])
    for domain in ("claude.ai", "chatgpt.com", "gemini.google.com", "grok.com"):
        assert domain in content_matches
