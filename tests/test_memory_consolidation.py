import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.memory_controller import MemoryController
from memory.memory_detector import MemoryDetector


def test_consolidation_preview_commit_and_noise_filtering(tmp_path, monkeypatch):
    monkeypatch.setenv("AIVCO_LOCAL_ENABLED", "false")
    monkeypatch.setenv("AIVCO_DATA_DIR", str(tmp_path / "data"))
    memory = MemoryController(tmp_path / "profile.json")
    memory.forget_all()
    detector = MemoryDetector()
    messages = ["My name is Alex.", "I am tired today.", "What is Python?"]
    preview = detector.consolidate_session(messages, memory, dry_run=True)
    assert len(preview["accepted"]) == 1
    assert len(preview["rejected"]) == 2
    assert memory.recall("name_alex") is None
    committed = detector.consolidate_session(messages, memory)
    assert committed["accepted"]
    assert memory.recall("name_alex") == "Alex"
    assert memory.get_record("name_alex")["provenance"]["source"] == "My name is Alex."


def test_manual_upsert_delete_keeps_semantic_index_in_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("AIVCO_LOCAL_ENABLED", "false")
    monkeypatch.setenv("AIVCO_DATA_DIR", str(tmp_path / "data"))
    memory = MemoryController(tmp_path / "profile.json")
    record = memory.upsert_memory("fact", "favorite_car", "I drive an automobile")
    assert record["value"] == "I drive an automobile"
    assert any(hit["memory_id"].endswith("favorite_car") for hit in memory.semantic_index.search("vehicle"))
    assert memory.delete_memory("fact", "favorite_car") is True
    assert not any(hit["memory_id"].endswith("favorite_car") for hit in memory.semantic_index.search("vehicle"))


def test_memory_summary_endpoint(monkeypatch):
    import controllers.llm_controller as llm_module
    monkeypatch.setattr(llm_module, "load_env_file", lambda env_path=".env": None)
    server = importlib.import_module("server.app")
    from fastapi.testclient import TestClient
    response = TestClient(server.app).get("/api/memory/summary")
    assert response.status_code == 200
    body = response.json()
    assert set(body["layers"]) == {"profile", "projects", "learned_facts", "episodes"}
    assert "rolling_backup_count" in body
    assert set(body["semantic_index"]) == {"healthy", "entry_count"}
