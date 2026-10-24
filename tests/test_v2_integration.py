"""
AIVCO V2 Integration Tests
Verifies full integration of MultiLayerMemory + route_and_generate
into CommandController and Aivco.main() loop.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.memory_controller import MemoryController
from memory.multi_layer_memory import MultiLayerMemory
from command_controller import CommandController
import Aivco
from llm_controller import LLMController, ModelGenerationError, ModelUnavailableError


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def v2_setup(tmp_path):
    """Returns (controller, tiered_memory) with isolated tmp paths."""
    flat_mem_path = tmp_path / "profile.json"
    v2_mem_path = str(tmp_path / "v2_memory.json")

    flat_mem = MemoryController(flat_mem_path)
    tiered_mem = MultiLayerMemory(storage_path=v2_mem_path)
    ctrl = CommandController(
        memory=flat_mem,
        confirm_input=lambda prompt: "yes",
        tiered_memory=tiered_mem,
    )
    return ctrl, tiered_mem


# ── 1. Core Profile CLI Commands ─────────────────────────────────────────────

class TestCoreCommands:
    def test_core_set_and_get(self, v2_setup):
        ctrl, mem = v2_setup
        result = ctrl.handle("core set primary_language = Python")
        assert "[Core] Saved: primary_language = Python" in result
        assert mem.get_core_fact("primary_language") == "Python"

    def test_core_set_value_with_spaces(self, v2_setup):
        ctrl, mem = v2_setup
        result = ctrl.handle("core set user_role = Principal AI Architect")
        assert "[Core] Saved: user_role = Principal AI Architect" in result
        assert mem.get_core_fact("user_role") == "Principal AI Architect"

    def test_core_get_existing(self, v2_setup):
        ctrl, mem = v2_setup
        mem.set_core_fact("os", "Windows")
        result = ctrl.handle("core get os")
        assert "[Core] os = Windows" in result

    def test_core_get_missing(self, v2_setup):
        ctrl, _ = v2_setup
        result = ctrl.handle("core get nonexistent_key")
        assert "[Core] Not found: nonexistent_key" in result

    def test_core_delete_existing(self, v2_setup):
        ctrl, mem = v2_setup
        mem.set_core_fact("temp_key", "temp_value")
        result = ctrl.handle("core delete temp_key")
        assert "[Core] Deleted: temp_key" in result
        assert mem.get_core_fact("temp_key") is None

    def test_core_delete_missing(self, v2_setup):
        ctrl, _ = v2_setup
        result = ctrl.handle("core delete nonexistent")
        assert "[Core] Not found: nonexistent" in result

    def test_core_persists_to_disk(self, tmp_path):
        """Verify set_core_fact actually writes to disk and reloads correctly."""
        v2_path = str(tmp_path / "v2_memory.json")
        flat_mem = MemoryController(tmp_path / "profile.json")
        mem1 = MultiLayerMemory(storage_path=v2_path)
        ctrl = CommandController(memory=flat_mem, tiered_memory=mem1)

        ctrl.handle("core set language = Python")

        # Reload from disk
        mem2 = MultiLayerMemory(storage_path=v2_path)
        assert mem2.get_core_fact("language") == "Python"

    def test_core_bad_usage(self, v2_setup):
        ctrl, _ = v2_setup
        result = ctrl.handle("core")
        assert "Usage" in result
        result2 = ctrl.handle("core unknown_subcommand value")
        assert "Usage" in result2


# ── 2. Project CLI Commands ───────────────────────────────────────────────────

class TestProjectCommands:
    def test_project_register_sets_active(self, v2_setup):
        ctrl, mem = v2_setup
        result = ctrl.handle("project register aivco AIVCO Second Brain")
        assert "aivco" in result
        assert "AIVCO Second Brain" in result
        assert ctrl.active_project_id == "aivco"
        assert "aivco" in mem.projects

    def test_project_use_switches_active(self, v2_setup):
        ctrl, mem = v2_setup
        mem.register_project("proj_a", "Project A")
        mem.register_project("proj_b", "Project B")
        ctrl.active_project_id = "proj_a"

        result = ctrl.handle("project use proj_b")
        assert "proj_b" in result
        assert ctrl.active_project_id == "proj_b"

    def test_project_use_unknown(self, v2_setup):
        ctrl, _ = v2_setup
        result = ctrl.handle("project use nonexistent")
        assert "Unknown project" in result

    def test_project_task_add(self, v2_setup):
        ctrl, mem = v2_setup
        mem.register_project("aivco", "AIVCO")
        result = ctrl.handle("project task add aivco Implement tiered memory")
        assert "added" in result.lower()
        assert "Implement tiered memory" in mem.projects["aivco"].pending_tasks

    def test_project_task_done(self, v2_setup):
        ctrl, mem = v2_setup
        mem.register_project("aivco", "AIVCO")
        mem.update_project_task("aivco", "Write tests", mark_completed=False)
        result = ctrl.handle("project task done aivco Write tests")
        assert "completed" in result.lower()
        assert "Write tests" not in mem.projects["aivco"].pending_tasks
        assert "Write tests" in mem.projects["aivco"].completed_milestones

    def test_project_task_unknown_project(self, v2_setup):
        ctrl, _ = v2_setup
        result = ctrl.handle("project task add unknown_proj Some task")
        assert "Unknown project" in result

    def test_project_task_no_duplicate_pending(self, v2_setup):
        ctrl, mem = v2_setup
        mem.register_project("p1", "P1")
        ctrl.handle("project task add p1 Deploy service")
        ctrl.handle("project task add p1 Deploy service")
        assert mem.projects["p1"].pending_tasks.count("Deploy service") == 1


# ── 3. Episode & Context CLI Commands ────────────────────────────────────────

class TestEpisodeAndContextCommands:
    def test_episode_add_logs_entry(self, v2_setup):
        ctrl, mem = v2_setup
        result = ctrl.handle("episode add Fixed CI pipeline | Pipeline green")
        assert "[Episode] Logged" in result
        assert len(mem.episodes) == 1
        assert mem.episodes[0].event == "Fixed CI pipeline"
        assert mem.episodes[0].outcome == "Pipeline green"

    def test_episode_links_to_active_project(self, v2_setup):
        ctrl, mem = v2_setup
        mem.register_project("aivco", "AIVCO")
        ctrl.active_project_id = "aivco"
        ctrl.handle("episode add Ran regression suite | 50 tests passed")
        assert mem.episodes[0].project_id == "aivco"

    def test_context_show_empty(self, v2_setup):
        ctrl, _ = v2_setup
        result = ctrl.handle("context show")
        assert "No V2 context available yet" in result or result == "[Context] No V2 context available yet."

    def test_context_show_all_three_layers(self, v2_setup):
        ctrl, mem = v2_setup
        # Populate all 3 layers
        mem.set_core_fact("user_role", "AI Engineer")
        mem.register_project("aivco", "AIVCO Brain", tech_stack=["Python"])
        mem.update_project_task("aivco", "Integrate V2 memory", mark_completed=False)
        mem.add_episode("Architecture review", "Approved", project_id="aivco")
        ctrl.active_project_id = "aivco"

        result = ctrl.handle("context show")
        assert "### Core Profile Context" in result
        assert "user_role: AI Engineer" in result
        assert "### Active Project: AIVCO Brain (aivco)" in result
        assert "Integrate V2 memory" in result
        assert "### Recent History / Lessons Learned" in result
        assert "Architecture review" in result

    def test_episode_add_without_pipe_defaults_outcome_to_logged(self, v2_setup):
        ctrl, mem = v2_setup
        result = ctrl.handle("episode add Read chapter 3 on neural networks")
        assert "[Episode] Logged" in result
        assert len(mem.episodes) == 1
        assert mem.episodes[0].event == "Read chapter 3 on neural networks"
        assert mem.episodes[0].outcome == "logged"

    def test_handle_strips_accidental_backticks(self, v2_setup):
        ctrl, mem = v2_setup
        result = ctrl.handle("`core set editor = VSCode`")
        assert "[Core] Saved: editor = VSCode" in result
        assert mem.get_core_fact("editor") == "VSCode"

    def test_episode_bad_usage(self, v2_setup):
        ctrl, _ = v2_setup
        result = ctrl.handle("episode add")
        assert "Usage" in result

        result2 = ctrl.handle("episode")
        assert "Usage" in result2


# ── 4. LLM Routing via Aivco.main() ──────────────────────────────────────────

class TestLLMRoutingIntegration:
    """Tests that Aivco.main() routes correctly using route_and_generate."""

    def _run_main(self, monkeypatch, tmp_path, inputs, route_side_effect=None, route_return=None):
        """Helper to run Aivco.main() with mocked inputs and LLM."""
        monkeypatch.setattr(Aivco, "MEMORY_FILE", tmp_path / "profile.json")
        monkeypatch.setattr(Aivco, "V2_MEMORY_FILE", str(tmp_path / "v2_memory.json"))

        route_calls = []

        class FakeRouterLLM:
            def __init__(self, *a, **kw):
                pass

            def is_available(self):
                return False

            def generate(self, sys_p, usr_p):
                return "local response"

            def route_and_generate(self, sys_p, usr_p, task_type="general"):
                route_calls.append({"system": sys_p, "user": usr_p, "task_type": task_type})
                if route_side_effect:
                    raise route_side_effect
                return route_return or {
                    "response": f"routed reply for: {usr_p}",
                    "provider_used": "groq",
                    "fallback_triggered": False,
                }

        monkeypatch.setattr(Aivco, "LLMController", FakeRouterLLM)
        it = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(it))

        history = Aivco.main()
        return history, route_calls

    def test_general_prompt_calls_route_and_generate(self, tmp_path, monkeypatch):
        history, calls = self._run_main(
            monkeypatch, tmp_path,
            inputs=["What is Python?", "exit"]
        )
        assert len(calls) == 1
        assert calls[0]["task_type"] == "general"
        assert calls[0]["user"] == "What is Python?"

    def test_deep_prefix_routes_to_deep_task_type(self, tmp_path, monkeypatch):
        history, calls = self._run_main(
            monkeypatch, tmp_path,
            inputs=["deep: Explain transformer architecture in depth", "exit"]
        )
        assert len(calls) == 1
        assert calls[0]["task_type"] == "deep"
        assert calls[0]["user"] == "Explain transformer architecture in depth"

    def test_private_prefix_routes_to_private_task_type(self, tmp_path, monkeypatch):
        history, calls = self._run_main(
            monkeypatch, tmp_path,
            inputs=["private: What is in my memory?", "exit"]
        )
        assert len(calls) == 1
        assert calls[0]["task_type"] == "private"
        assert calls[0]["user"] == "What is in my memory?"

    def test_v2_context_is_injected_into_system_prompt(self, tmp_path, monkeypatch):
        """Verify assembled V2 core profile context appears in the system_prompt passed to the LLM."""
        v2_path = str(tmp_path / "v2_memory.json")
        # Pre-populate V2 memory with core facts (core facts always appear regardless of active project)
        mem = MultiLayerMemory(storage_path=v2_path)
        mem.set_core_fact("expertise", "Machine Learning")
        mem.set_core_fact("hardware", "Windows Workstation")

        history, calls = self._run_main(
            monkeypatch, tmp_path,
            inputs=["What should I work on?", "exit"]
        )
        assert len(calls) == 1
        system_prompt = calls[0]["system"]
        # V2 core profile context must appear in the system prompt
        assert "V2 CONTEXT" in system_prompt
        assert "expertise: Machine Learning" in system_prompt
        assert "hardware: Windows Workstation" in system_prompt

    def test_route_and_generate_failure_falls_back_gracefully(self, tmp_path, monkeypatch, capsys):
        """When route_and_generate raises ModelUnavailableError, AIVCO shows the right message."""
        _, calls = self._run_main(
            monkeypatch, tmp_path,
            inputs=["Hello", "exit"],
            route_side_effect=ModelUnavailableError("No providers configured.")
        )
        captured = capsys.readouterr().out
        assert "[AIVCO] Local LLM is currently unavailable." in captured

    def test_route_and_generate_used_over_generate_when_available(self, tmp_path, monkeypatch):
        """Ensure route_and_generate is preferred over generate when it exists."""
        monkeypatch.setattr(Aivco, "MEMORY_FILE", tmp_path / "profile.json")
        monkeypatch.setattr(Aivco, "V2_MEMORY_FILE", str(tmp_path / "v2_memory.json"))

        generate_calls = []
        route_calls = []

        class FakeDualLLM:
            def __init__(self, *a, **kw):
                pass

            def is_available(self):
                return False

            def generate(self, sys_p, usr_p):
                generate_calls.append(usr_p)
                return "generate response"

            def route_and_generate(self, sys_p, usr_p, task_type="general"):
                route_calls.append(usr_p)
                return {"response": "routed response", "provider_used": "groq", "fallback_triggered": False}

        monkeypatch.setattr(Aivco, "LLMController", FakeDualLLM)
        it = iter(["Tell me a joke", "exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(it))
        Aivco.main()

        assert len(route_calls) == 1
        assert generate_calls == []  # generate() must NOT be called when route_and_generate exists
