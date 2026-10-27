"""
[DEPRECATED] scripts/launch_agent_browser.py
This script has been deprecated following the architecture pivot to
AIVCO Native Chrome Extension Bridge (Manifest V3).
Playwright CDP remote debugging on port 9222 is no longer required.
"""

def is_cdp_running() -> bool:
    """Backward compatibility stub returning Extension Bridge connectivity."""
    try:
        from browser_agent import extension_bridge_router
        return extension_bridge_router.is_connected()
    except Exception:
        return False

def launch_browser():
    print("[DEPRECATED] launch_agent_browser.py is no longer used.")
    print("AIVCO now uses the Native Chrome Extension Bridge (Manifest V3).")

if __name__ == "__main__":
    launch_browser()
