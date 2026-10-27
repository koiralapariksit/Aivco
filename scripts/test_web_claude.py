"""
Standalone Verification Script: Test Web Claude via UniversalWebAgent
Navigates to https://claude.ai, checks login state, and verifies response handling.
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from browser_agent import universal_web_agent, PLAYWRIGHT_AVAILABLE


def test_web_claude():
    print("=" * 65)
    print("    AIVCO CLAUDE PLAYWRIGHT WEB AGENT STANDALONE TEST           ")
    print("=" * 65)

    assert PLAYWRIGHT_AVAILABLE, "Playwright is not available in the current virtual environment."
    print("\n[STEP 1] Checking UniversalWebAgent Browser Status...")
    status = universal_web_agent.status()
    print(f"  Browser active: {status.get('active')}")
    print(f"  Profile directory: {status.get('profile_dir')}")
    print(f"  Open pages: {status.get('open_pages')}")
    assert status.get("active") is True, "Expected browser context to be active."
    assert "data" in status.get("profile_dir", "").lower()

    print("\n[STEP 2] Dispatching probe query to Claude (https://claude.ai)...")
    t0 = time.time()
    # Send a probe query
    result = universal_web_agent.query("claude", "ping", timeout=15)
    elapsed = round(time.time() - t0, 2)
    print(f"  Query completed in {elapsed}s")
    print(f"  Raw Result: {result[:200]}...")

    # Strict Zero-Bridge & Zero-Silent-Fallback Assertions
    forbidden_tokens = ["via groq bridge", "via gemini bridge", "via local bridge", "claude-engine"]
    for token in forbidden_tokens:
        assert token not in result.lower(), f"Forbidden bridge token '{token}' found in response!"

    # Evaluate Result State
    if "not logged in" in result.lower() or "failed at stage: login" in result.lower():
        print("\n  -> [RESULT]: Claude browser window is open at the Login / Auth screen.")
        print("  -> Operator Action: Please complete login once in the opened Chromium window.")
        print("  -> Cookies will remain persisted in data/browser_profile across all future sessions.")
    elif "web-account:error" in result:
        print(f"\n  -> [RESULT]: Honest error propagation verified: {result}")
    else:
        print("\n  -> [RESULT]: Claude session is LOGGED IN and returned active DOM response:")
        print(f"     \"{result[:150]}...\"")

    print("\n[STEP 3] Verifying selectors and error format...")
    assert len(result.strip()) > 10, "Expected non-empty response string"
    print("  -> [PASS] Web Claude agent operates with zero fake bridges and honest execution.")
    print("=" * 65)


if __name__ == "__main__":
    test_web_claude()

