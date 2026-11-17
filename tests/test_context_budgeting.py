import pytest

from kernel.cognitive_kernel import (
    TIER_CONTEXT_BUDGETS,
    budget_prompt_context,
    estimate_tokens,
)
from memory.memory_controller import MemoryController


def test_token_estimator_is_deterministic_and_handles_empty_values():
    assert estimate_tokens(None) == 0
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello, world!") == 6
    assert estimate_tokens("hello, world!") == estimate_tokens("hello, world!")


@pytest.mark.parametrize("tier", ["compact", "standard", "deep"])
def test_all_tiers_enforce_their_context_ceiling(tier):
    result = budget_prompt_context(
        system_prompt="System instructions must remain intact.",
        memory_context="Profile invariant: operator prefers concise responses.",
        conversation_history="\n\n".join([
            "OLD TURN " + "x " * (TIER_CONTEXT_BUDGETS[tier] + 100),
            "NEW TURN " + "y " * 100,
        ]),
        user_input="Current query stays intact.",
        tier=tier,
    )
    assert result["budget"] == TIER_CONTEXT_BUDGETS[tier]
    assert result["total_tokens"] <= result["budget"]
    assert result["components"]["system_prompt"] == "System instructions must remain intact."
    assert result["components"]["user_input"] == "Current query stays intact."
    assert "OLD TURN" not in result["components"]["conversation_history"]


def test_oldest_history_turn_is_dropped_before_memory():
    history = "\n\n".join([
        "OLD TURN " + "old " * 3_000,
        "NEW TURN " + "new " * 300,
    ])
    memory_context = "### Core Profile Context\n- Invariant: keep the operator profile."
    result = budget_prompt_context(
        system_prompt="System invariant.",
        memory_context=memory_context,
        conversation_history=history,
        user_input="Current query.",
        tier="compact",
    )
    components = result["components"]
    assert result["total_tokens"] <= result["budget"]
    assert "OLD TURN" not in components["conversation_history"]
    assert "NEW TURN" in components["conversation_history"]
    assert "Invariant: keep the operator profile." in components["memory_context"]
    assert components["system_prompt"] == "System invariant."
    assert components["user_input"] == "Current query."


def test_episodic_memory_is_trimmed_after_history_is_exhausted():
    memory_context = (
        "### Core Profile Context\n- Invariant: keep the operator profile.\n"
        "### Recent History / Lessons Learned\n"
        "- [2026-11-16] Event: " + "episode " * 4_000
    )
    result = budget_prompt_context(
        system_prompt="System invariant.",
        memory_context=memory_context,
        conversation_history="OLD TURN " + "history " * 100,
        user_input="Current query.",
        tier="compact",
    )
    components = result["components"]
    assert result["total_tokens"] <= result["budget"]
    assert components["conversation_history"] == ""
    assert "Invariant: keep the operator profile." in components["memory_context"]
    assert "[2026-11-16] Event:" not in components["memory_context"]


def test_retrieved_memories_use_memory_controller_token_budget(tmp_path):
    memory = MemoryController(tmp_path / "profile.json")
    memory.forget_all()
    memory.remember("favorite_language", "Python", "preference", "critical")
    memory.remember("old_project", "A long project detail " * 10, "project", "low")

    result = budget_prompt_context(
        system_prompt="S",
        memory_context="Profile invariant: preserve this.",
        user_input="Python question",
        tier="compact",
        memory_controller=memory,
    )
    assert result["total_tokens"] <= result["budget"]
    assert "Profile invariant: preserve this." in result["components"]["memory_context"]
    assert "favorite_language" in result["components"]["memory_context"]


def test_protected_text_over_budget_fails_instead_of_truncating():
    with pytest.raises(ValueError, match="System prompt and current user input"):
        budget_prompt_context(
            system_prompt="instruction " * 2_000,
            user_input="Current query.",
            tier="compact",
        )
