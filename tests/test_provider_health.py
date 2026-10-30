import sys
from pathlib import Path
import pytest
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel.arbiter import classify_failure, provider_health
from controllers.llm_controller import ModelUnavailableError, ModelGenerationError
from agents.browser_agent import universal_web_agent


def test_classify_failure_avoids_false_positives():
    """Verify classify_failure avoids false positives and detects specific failure patterns."""
    # False positive checks
    assert classify_failure("generation error") == "unavailable"
    assert classify_failure("rate of change is high") == "unavailable"
    assert classify_failure("generic network failure") == "unavailable"
    assert classify_failure("") == "unavailable"
    assert classify_failure(None) == "unavailable"

    # Rate limit checks
    assert classify_failure("Rate limit reached") == "rate_limit"
    assert classify_failure("RATE_LIMIT_REACHED") == "rate_limit"
    assert classify_failure("HTTP 429: Too Many Requests") == "rate_limit"
    assert classify_failure("Ratelimit active for 10 minutes") == "rate_limit"
    assert classify_failure("rate_limit_reached") == "rate_limit"

    # Timeout checks
    assert classify_failure("request timed out") == "timeout"
    assert classify_failure("Execution timeout exceeded 180s") == "timeout"


@pytest.mark.anyio
async def test_rate_limit_applies_600s_cooldown_both_paths(monkeypatch):
    """
    Verify rate-limit error on BOTH explicit Stage 0D path and auto web path
    sets ~600s cooldown without being overwritten by ~240s.
    """
    from server.app import _run_web_account, llm_controller

    try:
        provider_health.reset_all()

        async def mock_dispatch_rate_limit(*args, **kwargs):
            return "429: rate limit exceeded, wait a few minutes"

        monkeypatch.setattr(universal_web_agent, "dispatch_prompt_async", mock_dispatch_rate_limit)

        def mock_fallback_generate(*args, **kwargs):
            return {"response": "Fast mock fallback", "provider_used": "groq"}

        monkeypatch.setattr(llm_controller, "route_and_generate", mock_fallback_generate)

        # 1. Explicit web provider path (escalated=False)
        plan_explicit = {
            "action": "web_account",
            "provider": "claude",
            "clean_prompt": "build a landing page",
            "web_search": False,
            "escalated": False,
            "handshake": "Routing -> Claude",
            "initial_tier": "claude",
            "mode": "casual",
            "project_id": None,
        }
        res_explicit = await _run_web_account(plan_explicit, "use claude to build a landing page", time.time())
        assert "⚡ [aivco-kernel] CLAUDE web session unavailable/rate-limited" in res_explicit["response"]
        
        cd_explicit = provider_health.get_cooldown_remaining("claude")
        # Must be ~600s, NOT overwritten by 240s
        assert 580.0 <= cd_explicit <= 600.0, f"Expected ~600s cooldown, got {cd_explicit}"
        rec_claude = provider_health.get_record("claude")
        assert rec_claude.consecutive_failures == 1

        # 2. Auto-mode web path (escalated=True)
        provider_health.reset_all()
        plan_auto = {
            "action": "web_account",
            "provider": "chatgpt",
            "clean_prompt": "design an advanced portfolio",
            "web_search": False,
            "escalated": True,
            "handshake": "Arbiter -> ChatGPT",
            "initial_tier": "chatgpt",
            "mode": "casual",
            "project_id": None,
        }
        res_auto = await _run_web_account(plan_auto, "design an advanced portfolio", time.time())
        assert "⚡ [aivco-kernel] CHATGPT web session unavailable/rate-limited" in res_auto["response"]

        cd_auto = provider_health.get_cooldown_remaining("chatgpt")
        assert 580.0 <= cd_auto <= 600.0, f"Expected ~600s cooldown, got {cd_auto}"
        rec_chatgpt = provider_health.get_record("chatgpt")
        assert rec_chatgpt.consecutive_failures == 1

    finally:
        provider_health.reset_all()


def test_success_resets_consecutive_failures():
    """Verify consecutive_failures is reset to 0 upon recording a success."""
    try:
        provider_health.reset_all()
        provider_health.record_failure("groq", reason="timeout")
        provider_health.record_failure("groq", reason="timeout")
        rec = provider_health.get_record("groq")
        assert rec.consecutive_failures == 2
        assert rec.healthy is False

        provider_health.record_success("groq", latency_ms=85.0)
        assert rec.consecutive_failures == 0
        assert rec.healthy is True
        assert rec.cooldown_until == 0.0
        assert provider_health.get_cooldown_remaining("groq") == 0.0
    finally:
        provider_health.reset_all()


@pytest.mark.anyio
async def test_unknown_provider_failure_does_not_blame_groq(monkeypatch):
    """
    Simulate an auto-mode exception where neither exception nor request
    has an explicit provider. Verify provider_health.get_cooldown_remaining('groq') == 0.0.
    """
    from server.app import _run_llm, llm_controller

    try:
        provider_health.reset_all()

        # 1. ModelUnavailableError with no provider attribute
        def mock_unavailable(*args, **kwargs):
            raise ModelUnavailableError("Cluster unreachable")

        monkeypatch.setattr(llm_controller, "route_and_generate", mock_unavailable)

        plan = {
            "action": "llm",
            "provider": None,  # Auto-mode
            "task_type": "general",
            "private": False,
            "clean_prompt": "tell me a fact about space",
            "initial_tier": None,
            "handshake": "",
            "mode": "casual",
            "project_id": None,
        }

        res = await _run_llm(plan, "tell me a fact about space", time.time())
        assert "[AIVCO] Model unavailable: Cluster unreachable" in res["response"]
        assert provider_health.get_cooldown_remaining("groq") == 0.0
        assert provider_health.get_record("groq").consecutive_failures == 0

        # 2. ModelGenerationError with no provider attribute
        def mock_gen_err(*args, **kwargs):
            raise ModelGenerationError("Model output stream dropped")

        monkeypatch.setattr(llm_controller, "route_and_generate", mock_gen_err)
        res2 = await _run_llm(plan, "tell me a fact about space", time.time())
        assert "[AIVCO] Generation error: Model output stream dropped" in res2["response"]
        assert provider_health.get_cooldown_remaining("groq") == 0.0
        assert provider_health.get_record("groq").consecutive_failures == 0

        # 3. ModelGenerationError WITH provider attribute (e.g. gemini failed)
        err_with_provider = ModelGenerationError("Gemini quota 429")
        err_with_provider.provider = "gemini"

        def mock_gemini_err(*args, **kwargs):
            raise err_with_provider

        monkeypatch.setattr(llm_controller, "route_and_generate", mock_gemini_err)
        res3 = await _run_llm(plan, "tell me a fact about space", time.time())
        assert "[AIVCO] Generation error" in res3["response"]
        # Gemini must be degraded, Groq must remain completely unaffected!
        assert provider_health.get_cooldown_remaining("groq") == 0.0
        assert provider_health.get_record("groq").consecutive_failures == 0
        assert provider_health.get_cooldown_remaining("gemini") > 0.0
        assert provider_health.get_record("gemini").consecutive_failures == 1

    finally:
        provider_health.reset_all()


@pytest.mark.anyio
async def test_auto_web_success_and_fallback_health_reporting(monkeypatch):
    """
    Verify auto-mode web path and Stage 2 web fallback report success/failure properly.
    """
    from server.app import _run_web_account, _run_web_fallback

    try:
        provider_health.reset_all()

        # 1. Auto-mode web path success
        async def mock_success(*args, **kwargs):
            return "Here is your fully synthesized webpage."

        monkeypatch.setattr(universal_web_agent, "dispatch_prompt_async", mock_success)

        # Set 1 consecutive failure first
        provider_health.record_failure("claude", reason="timeout")
        assert provider_health.get_record("claude").consecutive_failures == 1

        plan_auto = {
            "action": "web_account",
            "provider": "claude",
            "clean_prompt": "make a website",
            "web_search": False,
            "escalated": True,
            "handshake": "",
            "initial_tier": "claude",
            "mode": "casual",
            "project_id": None,
        }
        res_auto = await _run_web_account(plan_auto, "make a website", time.time())
        assert "Here is your fully synthesized webpage." in res_auto["response"]
        # Success must reset consecutive_failures on claude
        assert provider_health.get_record("claude").consecutive_failures == 0
        assert provider_health.get_record("claude").healthy is True

        # 2. Stage 2 web fallback success
        monkeypatch.setattr(universal_web_agent, "query", lambda prov, prompt: "Fallback response content")
        plan_fb = {
            "action": "web_fallback",
            "provider": "chatgpt",
            "clean_prompt": "hello fallback",
            "initial_tier": None,
            "handshake": "",
            "mode": "casual",
            "project_id": None,
        }
        res_fb = await _run_web_fallback(plan_fb, "hello fallback", time.time())
        assert res_fb["response"] == "Fallback response content"
        assert provider_health.get_record("chatgpt").healthy is True

        # 3. Stage 2 web fallback failure
        monkeypatch.setattr(
            universal_web_agent,
            "query",
            lambda prov, prompt: "[aivco-kernel] Execution timed out waiting for AI response.",
        )
        res_fb_fail = await _run_web_fallback(plan_fb, "hello fallback", time.time())
        assert "[aivco-kernel] Execution timed out" in res_fb_fail["response"]
        assert provider_health.get_record("chatgpt").consecutive_failures == 1
        assert provider_health.get_record("chatgpt").last_failure_reason == "timeout"
        assert provider_health.get_cooldown_remaining("chatgpt") > 0.0

    finally:
        provider_health.reset_all()
