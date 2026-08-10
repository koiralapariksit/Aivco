import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Aivco
from command_controller import CommandController, HELP_TEXT
from memory.memory_controller import MemoryController


def make_controller(tmp_path):
    path = tmp_path / "profile.json"
    memory = MemoryController(path)
    return CommandController(memory, confirm_input=lambda prompt: "yes"), memory


def test_help_is_handled_as_a_command(tmp_path):
    controller, _ = make_controller(tmp_path)

    assert controller.handle("help") == HELP_TEXT
    assert controller.handle("HELP") == HELP_TEXT
    assert controller.handle("Help") == HELP_TEXT


def test_remember_parses_whitespace_and_saves_memory(tmp_path):
    controller, memory = make_controller(tmp_path)

    result = controller.handle("remember   name   =   Parikshit")

    assert result == "Saved: name = Parikshit"
    assert memory.recall("name") == "Parikshit"


def test_invalid_recall_is_reported_as_usage(tmp_path):
    controller, _ = make_controller(tmp_path)

    assert controller.handle("recall") == "Usage: recall <key>"
    assert controller.handle("recall =") == "Usage: recall <key>"
    assert controller.handle("recall name extra") == "Usage: recall <key>"


def test_invalid_forget_is_reported_as_usage(tmp_path):
    controller, _ = make_controller(tmp_path)

    assert controller.handle("forget") == "Usage: forget <key>   (or 'forget all' to clear everything)"
    assert controller.handle("forget =") == "Usage: forget <key>"


def test_normal_prompt_is_not_a_command(tmp_path):
    controller, _ = make_controller(tmp_path)

    assert controller.handle("hello") is None
    assert controller.handle("what is Python?") is None


def test_main_loop_routes_commands_to_controller_and_prompts_to_llm(tmp_path, monkeypatch, capsys):
    memory_path = tmp_path / "profile.json"
    monkeypatch.setattr(Aivco, "MEMORY_FILE", memory_path)

    llm_calls = []

    class FakeLLM:
        def __init__(self):
            pass

        def generate(self, system_prompt, user_prompt):
            llm_calls.append(user_prompt)

    monkeypatch.setattr(Aivco, "LLMController", FakeLLM)

    responses = iter(["help", "remember name = Parikshit", "recall name", "hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    Aivco.main()

    captured = capsys.readouterr().out
    assert "AIVCO commands:" in captured
    assert "Saved: name = Parikshit" in captured
    assert "name = Parikshit" in captured
    assert llm_calls == ["hello"]
