import json
import sys
from pathlib import Path

import httpx
import pytest
import memory.semantic_index as semantic_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.memory_controller import MemoryController
from memory.multi_layer_memory import MultiLayerMemory
from memory.semantic_index import SemanticIndex


def offline(monkeypatch, data_dir):
    monkeypatch.setenv("AIVCO_LOCAL_ENABLED", "false")
    monkeypatch.setenv("AIVCO_DATA_DIR", str(data_dir))


def test_offline_fallback_retrieves_paraphrase_with_synonyms(tmp_path, monkeypatch):
    offline(monkeypatch, tmp_path / "data")
    memory = MemoryController(tmp_path / "profile.json")
    memory.forget_all()
    memory.remember("transport", "A reliable automobile helps me commute", "fact", "high")
    memory.remember("music", "I enjoy writing songs", "interest", "medium")

    results = memory.search("I need a dependable car for travel")
    assert "transport" in results
    assert results["transport"]["value"] == "A reliable automobile helps me commute"


def test_local_embeddings_endpoint_and_graceful_timeout_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("AIVCO_LOCAL_ENABLED", "true")
    monkeypatch.setenv("AIVCO_LOCAL_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("AIVCO_EMBED_MODEL", "embed-test")
    monkeypatch.setitem(semantic_module._EMBED_FAILURE_UNTIL, "http://127.0.0.1:11434/v1", 0.0)
    index = SemanticIndex(tmp_path)
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout.connect))
        if json["input"] == "timeout case":
            raise httpx.ReadTimeout("simulated slow embedding server")
        return type("Response", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"data": [{"embedding": [3.0, 4.0]}]},
        })()

    monkeypatch.setattr("memory.semantic_index.httpx.post", fake_post)
    vector, mode = index.embed("online case")
    assert mode == "local:embed-test"
    assert vector == pytest.approx([0.6, 0.8])
    assert calls[0][0] == "http://127.0.0.1:11434/v1/embeddings"
    assert calls[0][1] == {"model": "embed-test", "input": "online case"}
    assert calls[0][2] == pytest.approx(0.4)

    fallback, fallback_mode = index.embed("timeout case")
    assert fallback_mode == "fallback:v1"
    assert len(fallback) == 384
    assert any(fallback)


def test_deleted_and_corrupt_sqlite_index_rebuild_from_json(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    offline(monkeypatch, data_dir)
    profile_path = tmp_path / "profile.json"
    memory = MemoryController(profile_path)
    memory.forget_all()
    memory.remember("career", "I hope to become a physician", "career", "high")
    index_path = data_dir / "semantic_index.db"
    assert index_path.exists()

    index_path.unlink()
    rebuilt_after_delete = MemoryController(profile_path)
    assert "career" in rebuilt_after_delete.search("doctor career")

    index_path.write_bytes(b"not a sqlite database")
    rebuilt_after_corruption = MemoryController(profile_path)
    assert "career" in rebuilt_after_corruption.search("medical doctor")
    assert json.loads(profile_path.read_text(encoding="utf-8"))["memories"]["career"]["value"] == "I hope to become a physician"


def test_token_budget_is_strict_for_semantically_retrieved_results(tmp_path, monkeypatch):
    offline(monkeypatch, tmp_path / "data")
    memory = MemoryController(tmp_path / "profile.json")
    memory.forget_all()
    memory.remember("car", "automobile vehicle transportation", "fact", "critical")
    memory.remember("buy", "purchase acquire shopping", "fact", "high")
    memory.remember("work", "career occupation profession", "fact", "medium")

    results = memory.search("vehicle purchase profession", token_budget=8, max_items=10)
    estimated_tokens = sum(max(1, (len(f"{key}: {record['value']}") + 3) // 4) for key, record in results.items())
    assert estimated_tokens <= 8
    assert len(results) <= 10

    sliced = memory.recall(token_budget=5, query="vehicle transportation")
    assert sum(max(1, (len(f"{key}: {value}") + 3) // 4) for key, value in sliced.items()) <= 5


def test_multilayer_episodes_and_backup_recovery_sync_into_index(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    offline(monkeypatch, data_dir)
    path = tmp_path / "multilayer.json"
    memory = MultiLayerMemory(str(path))
    memory.set_core_fact("career_goal", "Become a physician")
    memory.set_core_fact("study_plan", "Prepare for medical school")
    episode = memory.add_episode("Completed biology revision", "Strong exam readiness")
    hits = memory.semantic_index.search("doctor profession")
    assert any(hit["category"] == "core_profile" for hit in hits)
    assert any(hit["memory_id"].endswith(f"episode:{episode.id}") for hit in memory.semantic_index.search("study exam"))

    path.write_text("{truncated", encoding="utf-8")
    recovered = MultiLayerMemory(str(path))
    assert recovered.get_core_fact("career_goal") == "Become a physician"
    assert any(hit["category"] == "core_profile" for hit in recovered.semantic_index.search("doctor career"))
