import json
from concurrent.futures import ThreadPoolExecutor

from memory.memory_controller import MemoryController
from memory.multi_layer_memory import MultiLayerMemory


def test_default_storage_path_respects_data_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AIVCO_DATA_DIR", str(tmp_path / "memory-data"))
    memory = MultiLayerMemory()
    assert memory.storage_path == tmp_path / "memory-data" / "multilayer_profile.json"


def test_corrupt_or_truncated_primary_restores_newest_valid_backup(tmp_path):
    path = tmp_path / "profile.json"
    memory = MultiLayerMemory(str(path))
    memory.set_core_fact("favorite_language", "Python")
    memory.set_core_fact("current_project", "AIVCO")  # snapshots the prior valid state

    backups = list((tmp_path / ".backups").glob("profile.json.*.json"))
    assert backups
    path.write_text('{"core_profile":', encoding="utf-8")

    recovered = MultiLayerMemory(str(path))
    assert recovered.get_core_fact("favorite_language") == "Python"
    assert json.loads(path.read_text(encoding="utf-8"))["core_profile"]["favorite_language"]["value"] == "Python"


def test_missing_primary_restores_valid_backup(tmp_path):
    path = tmp_path / "profile.json"
    memory = MultiLayerMemory(str(path))
    memory.set_core_fact("role", "engineer")
    memory.set_core_fact("location", "Kathmandu")
    path.unlink()

    recovered = MultiLayerMemory(str(path))
    assert recovered.get_core_fact("role") == "engineer"
    assert path.exists()


def test_backup_ring_keeps_five_parseable_snapshots(tmp_path):
    path = tmp_path / "ring.json"
    memory = MultiLayerMemory(str(path))
    for revision in range(8):
        memory.set_core_fact("revision", str(revision))

    backups = list((tmp_path / ".backups").glob("ring.json.*.json"))
    assert len(backups) == 5
    assert all(isinstance(json.loads(backup.read_text(encoding="utf-8")), dict) for backup in backups)


def test_threaded_read_write_stress_preserves_every_episode(tmp_path):
    path = tmp_path / "concurrent.json"
    writers = [MultiLayerMemory(str(path)) for _ in range(8)]

    def write_batch(index):
        memory = writers[index]
        for item in range(12):
            memory.add_episode(f"writer-{index}-event-{item}", "ok")
            reader = MultiLayerMemory(str(path))
            assert isinstance(reader.episodes, list)

    with ThreadPoolExecutor(max_workers=len(writers)) as pool:
        list(pool.map(write_batch, range(len(writers))))

    persisted = json.loads(path.read_text(encoding="utf-8"))
    expected = {f"writer-{writer}-event-{item}" for writer in range(8) for item in range(12)}
    assert {episode["event"] for episode in persisted["episodes"]} == expected


def test_memory_controller_upserts_conflicts_and_slices(tmp_path):
    memory = MemoryController(tmp_path / "memory.json")
    memory.forget_all()
    assert memory.remember("preferred_name", "Alex", "identity", "high")[0]
    first = memory.get_record("preferred_name").copy()
    assert memory.remember("name", "Alex", "identity", "medium")[0]
    assert "preferred_name" not in memory.all_records()
    assert memory.recall("name") == "Alex"

    assert memory.remember("name", "Sam", "identity", "critical")[0]
    updated = memory.get_record("name")
    assert updated["value"] == "Sam"
    assert updated["created_at"] == first["created_at"]
    assert updated["updated_at"] >= first["updated_at"]
    assert len(memory.all_records(max_items=1)) == 1
    assert len(memory.recall(max_items=1)) == 1
    assert memory.all_records(token_budget=1) == {}
