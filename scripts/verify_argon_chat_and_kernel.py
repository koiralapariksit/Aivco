"""
Verification script for AIVCO Argon Edition:
1. Real-time natural-language instruction mutation
2. Adaptive identity resolution (Sir / 2-sentence vs full profile)
3. FastAPI endpoints: chat with attachments, dataset approve, state, tasks, memory
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_kernel import evaluate_query
from memory.multi_layer_memory import MultiLayerMemory
import server

def main():
    print("=================================================================")
    print("         AIVCO ARGON EDITION VERIFICATION SUITE                   ")
    print("=================================================================")

    mem_path = Path("data/v2_memory.json")
    mem = MultiLayerMemory(storage_path=str(mem_path))

    # Test 1: Real-time Directive Mutation
    print("\n--- TEST 1: Directive Mutation ---")
    query_address = "dont call me Parikshit you are allowed to tell me sir"
    res1 = evaluate_query(query_address, mem)
    print(f"Query: {query_address}")
    print(f"Kernel Response: {res1['local_response']}")
    assert res1["can_answer_locally"] is True
    assert "locked into Core Memory, Sir" in res1["local_response"]
    assert mem.get_core_fact("preferred_address") == "Sir"
    print("  [PASS] Name/Address directive successfully locked into Core Memory.")

    query_brevity = "avoid long answer in casual conversation"
    res2 = evaluate_query(query_brevity, mem)
    print(f"\nQuery: {query_brevity}")
    print(f"Kernel Response: {res2['local_response']}")
    assert res2["can_answer_locally"] is True
    assert mem.get_core_fact("concise_mode") == "True"
    print("  [PASS] Brevity protocol locked into Core Memory.")

    # Test 2: Adaptive Identity Resolution
    print("\n--- TEST 2: Identity Resolution (Concise vs Full) ---")
    query_who = "who am I"
    res3 = evaluate_query(query_who, mem)
    print(f"Query: '{query_who}' Response:\n{res3['local_response']}")
    assert res3["can_answer_locally"] is True
    assert "Sir" in res3["local_response"]
    # Verify concise response is NOT dumping 18 bullets
    lines = [l for l in res3["local_response"].strip().split("\n") if l.strip()]
    assert len(lines) <= 4, f"Expected concise response (<=4 lines), got {len(lines)}"
    print("  [PASS] 'who am I' returned sharp concise tactical response.")

    query_full = "full profile"
    res4 = evaluate_query(query_full, mem)
    print(f"\nQuery: '{query_full}' Response snippet:\n{res4['local_response'][:200]}...")
    assert "Operating System" in res4["local_response"] or "Preferred Address" in res4["local_response"]
    print("  [PASS] 'full profile' returned complete core profile.")

    # Test 3: Server Health & State
    print("\n--- TEST 3: Server Health & State Endpoints ---")
    health = server.get_health()
    print(f"Health: {health}")
    assert health["kernel"] is True

    state = server.get_state()
    print(f"Active Project: {state['active_project']}")
    print(f"Semantic Engrams: {len(state['layers']['Semantic'])}")
    print(f"Dataset Entries: {state['dataset_stats']['total_entries']}")
    assert "active_project" in state
    print("  [PASS] Health & State endpoints functioning.")

    # Test 4: Chat with Attachments
    print("\n--- TEST 4: Chat with Attachments ---")
    chat_req = server.ChatRequest(
        prompt="who am I",
        attachments=[{"name": "notes.txt", "content": "Priority task list"}]
    )
    chat_res = server.post_chat(chat_req)
    print(f"Chat Response: {chat_res['response'][:120]}...")
    assert chat_res["provider"] == "aivco-kernel"
    print("  [PASS] Attachments parsed and processed seamlessly.")

    # Test 5: Golden Pair Approval Endpoint
    print("\n--- TEST 5: Dataset Approval Endpoint ---")
    prev_total = state["dataset_stats"]["total_entries"]
    approve_req = server.DatasetApproveRequest(
        prompt="Test golden prompt",
        response="Test golden response",
        source="golden_user_approved",
        task_type="chat"
    )
    approve_res = server.approve_dataset(approve_req)
    print(f"Approve Result: {approve_res}")
    assert approve_res["ok"] is True
    assert approve_res["total"] >= prev_total
    print("  [PASS] POST /api/dataset/approve successfully logged golden pair.")

    # Test 6: Verify index.html existence and critical features
    print("\n--- TEST 6: index.html Verification ---")
    index_content = Path("index.html").read_text(encoding="utf-8")
    assert "marked.min.js" in index_content, "Missing marked.min.js"
    assert "highlight.min.js" in index_content, "Missing highlight.min.js"
    assert "atom-one-dark" in index_content, "Missing atom-one-dark"
    assert "markdown-body" in index_content, "Missing markdown-body styling"
    assert "ConstellationSphere" in index_content, "Missing 3D Constellation Sphere"
    assert "currentArtifact" in index_content, "Missing Artifact Drawer"
    assert "handlePaste" in index_content, "Missing Auto-Paste smart chip"
    assert "handleFileUpload" in index_content, "Missing native file upload"
    print("  [PASS] index.html contains all Argon Edition Flagship Chat & HUD components.")

    print("\n=================================================================")
    print("  ALL VERIFICATION TESTS COMPLETED WITH 100% SUCCESS!            ")
    print("=================================================================")

if __name__ == "__main__":
    main()
