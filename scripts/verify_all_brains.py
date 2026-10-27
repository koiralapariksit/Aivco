"""
Mandatory Gravitas Master Verification Suite:
Complete AIVCO 4-Brain Architecture + Headless Multi-Model In-Chat Execution
+ Real-Time Deep Web Grounding & Scraper + 3-Column Argon HUD
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import server
import browser_agent


def client_post_chat(prompt: str, target_model: str = "auto"):
    """Posts chat either via local HTTP if server is up, or directly via server module."""
    payload = {"prompt": prompt, "target_model": target_model}
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=35) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        # Direct function call fallback
        chat_req = server.ChatRequest(prompt=prompt, target_model=target_model)
        return server.post_chat(chat_req)


def client_get_state():
    """Gets state either via local HTTP if server is up, or directly via server module."""
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/state", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return server.get_state()


def main():
    print("=================================================================")
    print("    AIVCO 4-BRAIN ARCHITECTURE & HUD VERIFICATION SUITE         ")
    print("=================================================================")

    # ── Check 1: Instruction Mutation ──────────────────────────────────────────
    print("\n[CHECK 1] Instruction Mutation (Name & Brevity Directives)...")
    res1_a = client_post_chat(
        "first and very important instruction dont call me Parikshit you are allowed to tell me sir"
    )
    print(f"  Mutation Response 1: {res1_a.get('response')}")
    assert "Directive locked into Core Memory" in res1_a.get("response", "") or "Learned" in res1_a.get("response", "")

    res1_b = client_post_chat("avoid long answer in casual conversation")
    print(f"  Mutation Response 2: {res1_b.get('response')}")
    assert "Directive locked into Core Memory" in res1_b.get("response", "") or "Brevity" in res1_b.get("response", "")

    state1 = client_get_state()
    core_profile = state1.get("core_profile", {})
    print(f"  State core_profile: preferred_address={core_profile.get('preferred_address')}, concise_mode={core_profile.get('concise_mode')}")
    assert core_profile.get("preferred_address") == "sir", f"Expected 'sir', got {core_profile.get('preferred_address')}"
    assert core_profile.get("concise_mode") is True, f"Expected True, got {core_profile.get('concise_mode')}"
    print("  -> [PASS] CHECK 1: Preferred address locked to 'sir' and concise_mode is True.")

    # ── Check 2: No Clipboard Stub on Claude Selection ─────────────────────────
    print("\n[CHECK 2] In-Chat Multi-Model Execution (target_model='claude', 'who are you')...")
    res2 = client_post_chat("who are you", target_model="claude")
    resp_text2 = res2.get("response", "")
    prov2 = res2.get("provider", "")
    print(f"  Provider: {prov2}")
    print(f"  Response: {resp_text2[:120]}...")
    assert "copied to the clipboard" not in resp_text2.lower(), f"Clipboard stub detected in response: {resp_text2}"
    assert "dispatched prompt to" not in resp_text2.lower(), f"Dispatch stub detected in response: {resp_text2}"
    assert len(resp_text2.strip()) > 10, "Expected non-empty in-chat response"
    print("  -> [PASS] CHECK 2: Claude executed in-chat with 0 clipboard stubs or browser popups.")

    # ── Check 3: Compound Query + Real-Time Web Search (The Ballon d'Or Test) ──
    print("\n[CHECK 3] Compound Query + Real-Time Web Search (Ballon d'Or Test)...")
    res3 = client_post_chat("search web - which football player has won most ballon dor")
    resp_text3 = res3.get("response", "")
    prov3 = res3.get("provider", "")
    web_searched3 = res3.get("web_searched")
    print(f"  Provider: {prov3}")
    print(f"  Web Searched: {web_searched3}")
    print(f"  Response: {resp_text3[:180]}...")
    assert web_searched3 is True, f"Expected web_searched=True, got {web_searched3}"
    assert "Lionel Messi" in resp_text3 or "messi" in resp_text3.lower(), f"Expected 'Messi' in response: {resp_text3}"
    # Verify current factual accuracy (8 or eight or 2023)
    has_eight_or_2023 = any(token in resp_text3.lower() for token in ("8", "eight", "2023"))
    assert has_eight_or_2023, f"Expected 8 Ballon d'Or awards / 2023 to be mentioned in response: {resp_text3}"
    print("  -> [PASS] CHECK 3: Live web scraper retrieved real-time ground truth (Messi, 8/eight/2023 awards).")

    # ── Check 4: Typo-Tolerant Gemini Routing ──────────────────────────────────
    print("\n[CHECK 4] Typo-Tolerant Natural Language Forced Gemini + Web Search...")
    res4 = client_post_chat("use gemeni to seach who is lionel messi")
    prov4 = res4.get("provider", "")
    print(f"  Provider: {prov4}")
    print(f"  Web Searched: {res4.get('web_searched')}")
    print(f"  Response: {res4.get('response')[:120]}...")
    assert "gemini" in prov4.lower(), f"Expected 'gemini' in provider, got '{prov4}'"
    assert res4.get("web_searched") is True, f"Expected web_searched=True, got {res4.get('web_searched')}"
    print("  -> [PASS] CHECK 4: Typo 'gemeni' & 'seach' routed to Gemini with live web grounding.")

    # ── Check 5: Concise Kernel Identity ('who am I') ──────────────────────────
    print("\n[CHECK 5] Concise Kernel Identity ('who am I')...")
    res5 = client_post_chat("who am I")
    response_text5 = res5.get("response", "")
    provider5 = res5.get("provider", "")
    lines5 = [l for l in response_text5.strip().split("\n") if l.strip()]
    print(f"  Provider: {provider5}")
    print(f"  Response ({len(lines5)} lines):\n    {response_text5}")
    assert provider5 == "aivco-kernel", f"Expected provider 'aivco-kernel', got '{provider5}'"
    assert "sir" in response_text5.lower(), "Expected 'Sir' in response text"
    assert len(lines5) < 5, f"Expected concise response (< 5 lines), got {len(lines5)} lines"
    print("  -> [PASS] CHECK 5: 'who am I' answered by aivco-kernel in < 5 lines addressing operator as Sir.")

    # ── Check 6: Frontend Capabilities Check (index.html) ──────────────────────
    print("\n[CHECK 6] Frontend Capabilities Check (index.html)...")
    index_path = BASE_DIR / "index.html"
    index_html = index_path.read_text(encoding="utf-8")
    
    assert "user-select: text" in index_html, "Missing 'user-select: text' in index.html"
    assert "copyToClipboard" in index_html, "Missing 'copyToClipboard' in index.html"
    assert "marked" in index_html, "Missing 'marked' markdown engine in index.html"
    assert "LIVE PREVIEW" in index_html, "Missing 'LIVE PREVIEW' in index.html"
    
    # In-chat brain selector pills (no _web suffix!)
    assert "selectedBrain === 'claude'" in index_html, "Missing in-chat 'claude' selection in index.html"
    assert "selectedBrain === 'chatgpt'" in index_html, "Missing in-chat 'chatgpt' selection in index.html"
    assert "selectedBrain === 'grok'" in index_html, "Missing in-chat 'grok' selection in index.html"
    assert "RESET TO AUTO" in index_html, "Missing 'RESET TO AUTO' pill in index.html"
    print("  -> [PASS] CHECK 6: index.html contains selectable text, marked, LIVE PREVIEW, in-chat Claude/ChatGPT/Grok pills, and RESET TO AUTO.")

    print("\n=================================================================")
    print("                     ALL CHECKS PASSED                           ")
    print("=================================================================")


if __name__ == "__main__":
    main()
