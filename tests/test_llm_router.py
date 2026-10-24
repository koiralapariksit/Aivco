import os
from unittest.mock import patch, MagicMock
import pytest
from llm_controller import (
    LLMController,
    load_env_file,
    ModelUnavailableError,
    ModelGenerationError,
    AIVCOModelError,
)


def test_load_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Test comment\n"
        "\n"
        "TEST_KEY_ONE=value_one\n"
        'TEST_KEY_QUOTED="quoted_value"\n'
        "TEST_KEY_SINGLE='single_quoted'\n"
        "TEST_WITH_EQUALS=part1=part2\n",
        encoding="utf-8",
    )

    # Clean env before test
    for k in ("TEST_KEY_ONE", "TEST_KEY_QUOTED", "TEST_KEY_SINGLE", "TEST_WITH_EQUALS"):
        os.environ.pop(k, None)

    load_env_file(str(env_file))

    assert os.environ.get("TEST_KEY_ONE") == "value_one"
    assert os.environ.get("TEST_KEY_QUOTED") == "quoted_value"
    assert os.environ.get("TEST_KEY_SINGLE") == "single_quoted"
    assert os.environ.get("TEST_WITH_EQUALS") == "part1=part2"

    # Cleanup
    for k in ("TEST_KEY_ONE", "TEST_KEY_QUOTED", "TEST_KEY_SINGLE", "TEST_WITH_EQUALS"):
        os.environ.pop(k, None)


def test_route_and_generate_general_calls_groq_first():
    ctrl = LLMController()
    with patch.object(ctrl, "_generate_groq", return_value="Groq response") as mock_groq, \
         patch.object(ctrl, "_generate_gemini") as mock_gemini, \
         patch.object(ctrl, "generate") as mock_llama:
        res = ctrl.route_and_generate("sys prompt", "user prompt", task_type="general")
        assert res["provider_used"] == "groq"
        assert res["response"] == "Groq response"
        assert res["fallback_triggered"] is False
        mock_groq.assert_called_once_with("sys prompt", "user prompt")
        mock_gemini.assert_not_called()
        mock_llama.assert_not_called()


def test_route_and_generate_deep_calls_gemini_first():
    ctrl = LLMController()
    with patch.object(ctrl, "_generate_gemini", return_value="Gemini response") as mock_gemini, \
         patch.object(ctrl, "_generate_groq") as mock_groq, \
         patch.object(ctrl, "generate") as mock_llama:
        res = ctrl.route_and_generate("sys prompt", "user prompt", task_type="deep")
        assert res["provider_used"] == "gemini"
        assert res["response"] == "Gemini response"
        assert res["fallback_triggered"] is False
        mock_gemini.assert_called_once_with("sys prompt", "user prompt")
        mock_groq.assert_not_called()
        mock_llama.assert_not_called()


def test_route_and_generate_large_prompt_calls_gemini_first():
    ctrl = LLMController()
    sys_prompt = "s" * 3001
    user_prompt = "u" * 3001
    with patch.object(ctrl, "_generate_gemini", return_value="Gemini large response") as mock_gemini, \
         patch.object(ctrl, "_generate_groq") as mock_groq, \
         patch.object(ctrl, "generate") as mock_llama:
        res = ctrl.route_and_generate(sys_prompt, user_prompt, task_type="chat")
        assert res["provider_used"] == "gemini"
        assert res["response"] == "Gemini large response"
        assert res["fallback_triggered"] is False
        mock_gemini.assert_called_once_with(sys_prompt, user_prompt)
        mock_groq.assert_not_called()
        mock_llama.assert_not_called()


def test_route_and_generate_private_lock_never_calls_cloud():
    ctrl = LLMController()
    with patch.object(ctrl, "generate", return_value="Local response") as mock_llama, \
         patch.object(ctrl, "_generate_groq") as mock_groq, \
         patch.object(ctrl, "_generate_gemini") as mock_gemini:
        res = ctrl.route_and_generate("sys", "user", task_type="private")
        assert res["provider_used"] == "local_llama"
        assert res["response"] == "Local response"
        assert res["fallback_triggered"] is False
        mock_llama.assert_called_once_with("sys", "user")
        mock_groq.assert_not_called()
        mock_gemini.assert_not_called()

    # Verify that if local_llama fails under private lock, it raises and never falls back
    with patch.object(ctrl, "generate", side_effect=ModelGenerationError("local error")) as mock_llama, \
         patch.object(ctrl, "_generate_groq") as mock_groq, \
         patch.object(ctrl, "_generate_gemini") as mock_gemini:
        with pytest.raises(ModelGenerationError):
            ctrl.route_and_generate("sys", "user", task_type="private")
        mock_groq.assert_not_called()
        mock_gemini.assert_not_called()


def test_route_and_generate_groq_failure_falls_back_to_gemini():
    ctrl = LLMController()
    with patch.object(ctrl, "_generate_groq", side_effect=ModelGenerationError("Groq 500 error")) as mock_groq, \
         patch.object(ctrl, "_generate_gemini", return_value="Gemini fallback response") as mock_gemini, \
         patch.object(ctrl, "generate") as mock_llama:
        res = ctrl.route_and_generate("sys", "user", task_type="general")
        assert res["provider_used"] == "gemini"
        assert res["response"] == "Gemini fallback response"
        assert res["fallback_triggered"] is True
        mock_groq.assert_called_once_with("sys", "user")
        mock_gemini.assert_called_once_with("sys", "user")
        mock_llama.assert_not_called()


def test_route_and_generate_all_cloud_fail_falls_back_to_local_llama():
    ctrl = LLMController()
    with patch.object(ctrl, "_generate_groq", side_effect=ModelUnavailableError("No groq key")), \
         patch.object(ctrl, "_generate_gemini", side_effect=ModelGenerationError("Gemini quota")), \
         patch.object(ctrl, "generate", return_value="Local llama fallback") as mock_llama:
        res = ctrl.route_and_generate("sys", "user", task_type="general")
        assert res["provider_used"] == "local_llama"
        assert res["response"] == "Local llama fallback"
        assert res["fallback_triggered"] is True
        mock_llama.assert_called_once_with("sys", "user")


def test_route_and_generate_all_fail_raises():
    ctrl = LLMController()
    with patch.object(ctrl, "_generate_groq", side_effect=ModelUnavailableError("No groq key")), \
         patch.object(ctrl, "_generate_gemini", side_effect=ModelGenerationError("Gemini quota")), \
         patch.object(ctrl, "generate", side_effect=ModelUnavailableError("No local binary")):
        with pytest.raises(AIVCOModelError):
            ctrl.route_and_generate("sys", "user", task_type="general")

