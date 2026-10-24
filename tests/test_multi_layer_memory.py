import os
import pytest
from memory.multi_layer_memory import MultiLayerMemory, CoreFact, Episode, ProjectState

TEST_STORAGE = "data/test_multilayer.json"

@pytest.fixture
def memory_fixture(tmp_path):
    storage_file = str(tmp_path / "test_multilayer.json")
    mem = MultiLayerMemory(storage_path=storage_file)
    yield mem


def test_core_profile_lifecycle(tmp_path):
    storage_file = str(tmp_path / "test_lifecycle.json")
    mem = MultiLayerMemory(storage_path=storage_file)

    # 1. Set and get
    fact = mem.set_core_fact("primary_language", "Python", importance="critical")
    assert isinstance(fact, CoreFact)
    assert fact.key == "primary_language"
    assert fact.value == "Python"
    assert fact.importance == "critical"
    assert mem.get_core_fact("primary_language") == "Python"

    # 2. Overwrite
    mem.set_core_fact("primary_language", "Rust")
    assert mem.get_core_fact("primary_language") == "Rust"

    # 3. Disk reload verification
    reloaded = MultiLayerMemory(storage_path=storage_file)
    assert reloaded.get_core_fact("primary_language") == "Rust"

    # 4. Delete existing and non-existing
    assert reloaded.delete_core_fact("primary_language") is True
    assert reloaded.get_core_fact("primary_language") is None
    assert reloaded.delete_core_fact("non_existent_key") is False

    # 5. Verify deletion persists
    reloaded_after_delete = MultiLayerMemory(storage_path=storage_file)
    assert reloaded_after_delete.get_core_fact("primary_language") is None


def test_episodic_memory_filtering(tmp_path):
    storage_file = str(tmp_path / "test_episodes.json")
    mem = MultiLayerMemory(storage_path=storage_file)

    # Add episodes with different timestamps / projects
    ep1 = mem.add_episode(
        event="Setup CI/CD pipeline",
        outcome="Passing",
        tags=["devops", "ci"],
        project_id="aivco",
    )
    ep2 = mem.add_episode(
        event="Added unit tests for memory detector",
        outcome="All passed",
        tags=["testing"],
        project_id="aivco",
    )
    ep3 = mem.add_episode(
        event="Configured secondary research workspace",
        outcome="Connected",
        tags=["infra"],
        project_id="other_project",
    )

    assert len(ep1.id) == 8
    assert isinstance(ep1, Episode)

    # Retrieve all recent episodes newest-first
    all_recent = mem.get_recent_episodes(limit=5)
    assert len(all_recent) == 3
    # Check descending order by timestamp
    assert all_recent[0].timestamp >= all_recent[1].timestamp >= all_recent[2].timestamp

    # Filter by project_id
    aivco_eps = mem.get_recent_episodes(limit=5, project_id="aivco")
    assert len(aivco_eps) == 2
    assert all(e.project_id == "aivco" for e in aivco_eps)

    other_eps = mem.get_recent_episodes(limit=5, project_id="other_project")
    assert len(other_eps) == 1
    assert other_eps[0].event == "Configured secondary research workspace"

    # Limit check
    limited = mem.get_recent_episodes(limit=1, project_id="aivco")
    assert len(limited) == 1


def test_project_task_state_machine(tmp_path):
    storage_file = str(tmp_path / "test_projects.json")
    mem = MultiLayerMemory(storage_path=storage_file)

    # Register project
    proj = mem.register_project("aivco", "AIVCO V2", tech_stack=["Python", "Pytest"])
    assert isinstance(proj, ProjectState)
    assert proj.project_id == "aivco"
    assert proj.title == "AIVCO V2"
    assert proj.status == "active"
    assert proj.tech_stack == ["Python", "Pytest"]
    assert proj.pending_tasks == []
    assert proj.completed_milestones == []

    # Add pending tasks
    mem.update_project_task("aivco", "Implement MultiLayerMemory", mark_completed=False)
    mem.update_project_task("aivco", "Write unit test suite", mark_completed=False)
    # Duplicate pending task should not duplicate
    mem.update_project_task("aivco", "Implement MultiLayerMemory", mark_completed=False)

    proj_state = mem.projects["aivco"]
    assert proj_state.pending_tasks == ["Implement MultiLayerMemory", "Write unit test suite"]

    # Mark completed: should move from pending to completed_milestones
    mem.update_project_task("aivco", "Implement MultiLayerMemory", mark_completed=True)
    assert "Implement MultiLayerMemory" not in proj_state.pending_tasks
    assert "Implement MultiLayerMemory" in proj_state.completed_milestones

    # Duplicate mark completed
    mem.update_project_task("aivco", "Implement MultiLayerMemory", mark_completed=True)
    assert proj_state.completed_milestones.count("Implement MultiLayerMemory") == 1

    # Mark completed for task that wasn't in pending
    mem.update_project_task("aivco", "Direct completed milestone", mark_completed=True)
    assert "Direct completed milestone" in proj_state.completed_milestones

    # Unknown project should raise KeyError
    with pytest.raises(KeyError):
        mem.update_project_task("unknown_proj", "Any task")

    # Reload check
    reloaded = MultiLayerMemory(storage_path=storage_file)
    assert "Implement MultiLayerMemory" in reloaded.projects["aivco"].completed_milestones
    assert "Write unit test suite" in reloaded.projects["aivco"].pending_tasks


def test_context_assembly_output(tmp_path):
    storage_file = str(tmp_path / "test_context.json")
    mem = MultiLayerMemory(storage_path=storage_file)

    # Empty context returns empty string
    empty_context = mem.assemble_context()
    assert empty_context == ""

    # Populate all 3 layers
    mem.set_core_fact("user_role", "Principal AI Architect")
    mem.set_core_fact("hardware", "Windows Local-First Workstation")
    mem.register_project("aivco", "AIVCO Second Brain", tech_stack=["Python", "CLI"])
    mem.update_project_task("aivco", "Deploy V2 Tiered Architecture", mark_completed=False)
    mem.update_project_task("aivco", "V1.3 Baseline Verification", mark_completed=True)
    mem.add_episode(
        event="Ran test regression suite",
        outcome="37 tests passing",
        tags=["tests"],
        project_id="aivco",
    )

    context = mem.assemble_context(active_project_id="aivco", max_episodes=2)

    # Verify Layer 1
    assert "### Core Profile Context" in context
    assert "- user_role: Principal AI Architect" in context
    assert "- hardware: Windows Local-First Workstation" in context

    # Verify Layer 3
    assert "### Active Project: AIVCO Second Brain (aivco)" in context
    assert "- Status: active" in context
    assert "- Stack: Python, CLI" in context
    assert "- Completed: 1 milestones" in context
    assert "- Pending: Deploy V2 Tiered Architecture" in context

    # Verify Layer 2
    assert "### Recent History / Lessons Learned" in context
    assert "Event: Ran test regression suite | Outcome: 37 tests passing" in context


def test_corrupt_data_resilience(tmp_path):
    storage_file = str(tmp_path / "corrupt_data.json")

    # 1. Non-existent file starts clean without error
    mem = MultiLayerMemory(storage_path=storage_file)
    assert mem.core_profile == {}
    assert mem.episodes == []
    assert mem.projects == {}

    # 2. Corrupt/Invalid JSON
    with open(storage_file, "w", encoding="utf-8") as f:
        f.write("{ invalid json corrupted content !@#$")

    mem_corrupt = MultiLayerMemory(storage_path=storage_file)
    assert mem_corrupt.core_profile == {}
    assert mem_corrupt.episodes == []
    assert mem_corrupt.projects == {}

    # 3. Invalid data schema (e.g. list instead of dict, or invalid types)
    with open(storage_file, "w", encoding="utf-8") as f:
        f.write('{"version": "2.0-tiered", "core_profile": "not_a_dict"}')

    mem_invalid_schema = MultiLayerMemory(storage_path=storage_file)
    assert mem_invalid_schema.core_profile == {}
