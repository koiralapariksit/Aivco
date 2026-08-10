"""
AIVCO -- local personal AI assistant. Main entry point.

Architecture:

    user input
        |
        v
    CommandController.handle()
        |
        +-- command   -> execute directly in Python, print result, loop
        +-- not a command -> LLMController.generate()

Commands (remember/recall/forget/help/clear/exit) are always resolved
in CommandController + MemoryController, entirely in Python. The only
subprocess call in the whole program lives in LLMController, and it
is only ever reached for genuine questions -- never for commands,
malformed or otherwise.
"""

import json
import os
from pathlib import Path

from memory.memory_controller import MemoryController
from command_controller import CommandController, EXIT_SENTINEL, CLEAR_SENTINEL
from llm_controller import LLMController

BASE_DIR = Path(__file__).resolve().parent
MEMORY_FILE = BASE_DIR / "memory" / "user" / "profile.json"

# Location used by the earlier prototype. If it exists and the new
# location doesn't, migrate it automatically so old memories aren't lost.
OLD_MEMORY_FILE = BASE_DIR / "memory" / "user_profile.json"


def migrate_old_memory() -> None:
    if OLD_MEMORY_FILE.exists() and not MEMORY_FILE.exists():
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        OLD_MEMORY_FILE.replace(MEMORY_FILE)
        print(f"[AIVCO] Migrated existing memory to {MEMORY_FILE}")


def build_system_prompt(memory: dict) -> str:
    memory_text = json.dumps(memory, indent=2, ensure_ascii=False)

    return f"""
You are AIVCO, a personal AI assistant.

Your name is AIVCO.

Your purpose is to become the user's second brain.

You help with:
- programming
- studying
- research
- computer tasks
- planning
- projects
- organizing information
- general questions

You are concise, intelligent, practical and direct.

Never claim to be human.
Never call yourself a customer-support representative.

USER MEMORY:

{memory_text}

Use the memory when relevant.

Never invent memories.

When asked your name, answer:
"I am AIVCO."
"""


def main():
    migrate_old_memory()

    memory = MemoryController(MEMORY_FILE)
    commands = CommandController(memory)
    llm = LLMController()

    print("Starting AIVCO...")
    print("Type 'help' to see available commands.")
    print()

    while True:
        try:
            user_input = input("AIVCO > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAIVCO shutting down.")
            break

        if not user_input:
            continue

        result = commands.handle(user_input)

        if result is not None:
            if result == EXIT_SENTINEL:
                print("AIVCO shutting down.")
                break
            if result == CLEAR_SENTINEL:
                os.system("cls" if os.name == "nt" else "clear")
                continue
            if result:
                print(f"\n{result}\n")
            continue

        # Not a command -> this, and only this, path talks to the model.
        system_prompt = build_system_prompt(memory.as_dict())
        llm.generate(system_prompt, user_input)


if __name__ == "__main__":
    main()