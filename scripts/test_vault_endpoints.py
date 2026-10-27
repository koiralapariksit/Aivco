import urllib.request
import json
import sys

def main():
    print("Testing Vault Endpoints...")

    # 1. Test GET /api/vault/status
    req = urllib.request.Request("http://127.0.0.1:8000/api/vault/status")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert "claude_vault" in data, "Missing claude_vault"
        slots = data["claude_vault"]["slots"]
        assert len(slots) == 20, f"Expected 20 slots, got {len(slots)}"
        assert "chatgpt" in data["providers"], "Missing chatgpt in providers"
        assert "gemini" in data["providers"], "Missing gemini in providers"
        assert "grok" in data["providers"], "Missing grok in providers"
        print(f"[PASS] GET /api/vault/status: {len(slots)} Claude slots & 3 providers verified.")

    # 2. Test POST /api/vault/set_active_slot
    req2 = urllib.request.Request(
        "http://127.0.0.1:8000/api/vault/set_active_slot",
        data=json.dumps({"slot": 2}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req2) as resp:
        data2 = json.loads(resp.read().decode("utf-8"))
        assert data2.get("ok") is True
        assert data2.get("active_slot") == 2
        print("[PASS] POST /api/vault/set_active_slot: Slot 2 set as active.")

    # 3. Test Reset back to slot 1
    req3 = urllib.request.Request(
        "http://127.0.0.1:8000/api/vault/set_active_slot",
        data=json.dumps({"slot": 1}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req3) as resp:
        data3 = json.loads(resp.read().decode("utf-8"))
        assert data3.get("active_slot") == 1
        print("[PASS] POST /api/vault/set_active_slot: Reset active slot to 1.")

    # 4. Test POST /api/vault/update_credentials
    req4 = urllib.request.Request(
        "http://127.0.0.1:8000/api/vault/update_credentials",
        data=json.dumps({"provider": "claude", "slot": 1, "email": "operator_test@aivco.org"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req4) as resp:
        data4 = json.loads(resp.read().decode("utf-8"))
        assert data4.get("ok") is True
        print("[PASS] POST /api/vault/update_credentials: Email updated on slot 1.")

    # Verify update reflected in status
    req5 = urllib.request.Request("http://127.0.0.1:8000/api/vault/status")
    with urllib.request.urlopen(req5) as resp:
        data5 = json.loads(resp.read().decode("utf-8"))
        slot1 = data5["claude_vault"]["slots"][0]
        assert slot1["email"] == "operator_test@aivco.org", f"Expected email in slot 1, got {slot1['email']}"
        print("[PASS] GET /api/vault/status: Verified updated email in slot 1.")

    print("\nALL VAULT ENDPOINTS FUNCTIONING 100%!")

if __name__ == "__main__":
    main()

