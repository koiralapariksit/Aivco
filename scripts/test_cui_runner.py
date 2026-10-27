"""
Diagnostic test script for CUI runner health check.
Verifies:
1. Memory Layer Integrity (data/v2_memory.json)
2. CommandController CLI Handlers
3. LLM Router Prompt Routing (dry run / ping)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from memory.multi_layer_memory import MultiLayerMemory
from command_controller import CommandController
from llm_controller import LLMController


def run_test_1_memory_layer_integrity():
    print("\n--- Test 1: Memory Layer Integrity ---")
    storage_path = str(BASE_DIR / "data" / "v2_memory.json")
    mem = MultiLayerMemory(storage_path=storage_path)

    active_proj = mem.get_active_project()
    assert active_proj == "aivco", f"Expected active project 'aivco', got: {active_proj}"

    proj_state = mem.projects.get(active_proj)
    assert proj_state is not None, f"Active project '{active_proj}' not found in mem.projects"
    assert hasattr(proj_state, "completed_milestones") and len(proj_state.completed_milestones) > 0, "No completed milestones"
    assert hasattr(proj_state, "pending_tasks") and len(proj_state.pending_tasks) > 0, "No pending tasks"

    recent_episodes = mem.get_recent_episodes(limit=5)
    assert isinstance(recent_episodes, list), f"Expected list of episodes, got: {type(recent_episodes)}"
    assert len(recent_episodes) > 0, "Recent episodes list is empty"

    print("[PASS] Memory V2 & Active Project Verified.")
    return mem


def run_test_2_command_controller_handling(mem=None):
    print("\n--- Test 2: CommandController Handling ---")
    if mem is None:
        storage_path = str(BASE_DIR / "data" / "v2_memory.json")
        mem = MultiLayerMemory(storage_path=storage_path)

    cmd = CommandController(memory=mem, tiered_memory=mem)

    status_res = cmd.handle("project status")
    assert status_res is not None, "Expected string response from 'project status', got None"
    assert "aivco" in status_res.lower(), f"Unexpected project status output: {status_res}"

    # Verify project list
    list_res = cmd.handle("project list")
    assert list_res is not None, "Expected string response from 'project list', got None"

    # Verify episode add
    ep_add_res = cmd.handle("episode add Test Event | Test Outcome")
    assert ep_add_res is not None, "Expected string response from 'episode add', got None"

    print("[PASS] CommandController CLI Handlers Verified.")
    return cmd


def run_test_3_llm_router_prompt_routing():
    print("\n--- Test 3: LLM Router Dry Run ---")
    llm = LLMController()

    local_ready = llm.is_available()
    groq_ready = bool(os.environ.get("GROQ_API_KEY", "").strip())
    gemini_ready = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    print(f"  Tier 1 (Local):  {'Ready' if local_ready else 'Not available'}")
    print(f"  Tier 2 (Groq):   {'Ready' if groq_ready else 'Not configured'}")
    print(f"  Tier 3 (Gemini): {'Ready' if gemini_ready else 'Not configured'}")

    assert groq_ready or gemini_ready or local_ready, "No LLM tier is ready/configured!"

    # Run short test prompt
    prompt = "test memory ping"
    try:
        res = llm.route_and_generate(
            system_prompt="You are a test ping responder. Reply with a short one-sentence acknowledgement.",
            user_prompt=prompt,
            task_type="default",
        )
        response_text = res.get("response", "")
        provider_used = res.get("provider_used", "unknown")
        assert isinstance(response_text, str) and len(response_text.strip()) > 0, "Empty response from LLM"
        print(f"  Routed via: {provider_used}")
        print(f"  Response preview: {response_text.strip()[:60]}...")
        print("[PASS] LLM Routing & In-Memory Context Assembly Verified.")
    except Exception as e:
        print(f"[FAIL] LLM Routing failed with error: {e}")
        raise


if __name__ == "__main__":
    try:
        mem = run_test_1_memory_layer_integrity()
        cmd = run_test_2_command_controller_handling(mem)
        run_test_3_llm_router_prompt_routing()
        print("\n==============================================")
        print("ALL CUI DIAGNOSTICS COMPLETED SUCCESSFULLY!")
        print("==============================================")
    except Exception as exc:
        print(f"\n[DIAGNOSTIC FAILURE] {exc}")
        sys.exit(1)

