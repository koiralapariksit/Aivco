"""
AIVCO Command Controller

Decides whether a line of input is an AIVCO command (remember,
recall, forget, help, clear, exit, memory) or a normal question that
should go to the LLM. Every command is handled directly in Python --
never via subprocess -- and malformed commands are always caught
here, so they can never leak through to the model or the CLI.
"""

from typing import Optional

from memory.memory_controller import MemoryController

HELP_TEXT = """AIVCO commands:
  remember <key> = <value>   Save a memory
  recall [key]                Show one memory, or all memories
  memory                      Alias for 'recall'
  forget <key>                 Delete one memory
  forget all                   Delete ALL memories (asks to confirm)
  clear                        Clear the screen
  help                         Show this message
  exit                         Quit AIVCO

Anything else is sent to the AI model."""

EXIT_SENTINEL = "__EXIT__"
CLEAR_SENTINEL = "__CLEAR__"


class CommandController:
    def __init__(self, memory: MemoryController, confirm_input=input):
        self.memory = memory
        # Injectable so it can be swapped out in tests.
        self._confirm_input = confirm_input

    def _split_command(self, text: str) -> tuple[str, str]:
        parts = text.strip().split(None, 1)
        if not parts:
            return "", ""
        keyword = parts[0].lower()
        remainder = parts[1].strip() if len(parts) > 1 else ""
        return keyword, remainder

    def _classify(self, text: str) -> Optional[str]:
        keyword, remainder = self._split_command(text)

        if keyword == "":
            return "empty"
        if keyword == "exit":
            return "exit"
        if keyword == "help":
            return "help"
        if keyword == "clear":
            return "clear"
        if keyword == "memory":
            return "recall_all"
        if keyword == "recall":
            if not remainder:
                return "recall_bad"
            tokens = remainder.split()
            if len(tokens) != 1 or tokens[0] == "=":
                return "recall_bad"
            return "recall_one"
        if keyword == "forget":
            if not remainder:
                return "forget_bad"
            tokens = remainder.split()
            if len(tokens) == 1 and tokens[0].lower() == "all":
                return "forget_all"
            if len(tokens) != 1 or tokens[0] == "=":
                return "forget_bad"
            return "forget_one"
        if keyword == "remember":
            if not remainder:
                return "remember_bad"
            if "=" not in remainder:
                return "remember_bad"
            key, value = remainder.split("=", 1)
            if not key.strip() or not value.strip():
                return "remember_bad"
            return "remember"

        return None

    def is_command(self, raw_input: str) -> bool:
        return self._classify(raw_input.strip()) is not None

    def handle(self, raw_input: str) -> Optional[str]:
        """
        Returns:
          - None            -> not a command, send raw_input to the LLM
          - ""               -> command handled, nothing to print (e.g. blank line)
          - "__EXIT__"       -> caller should shut down
          - "__CLEAR__"      -> caller should clear the screen
          - any other string -> command handled, print this message
        """
        text = raw_input.strip()
        kind = self._classify(text)

        if kind is None:
            return None
        if kind == "empty":
            return ""
        if kind == "exit":
            return EXIT_SENTINEL
        if kind == "help":
            return HELP_TEXT
        if kind == "clear":
            return CLEAR_SENTINEL

        keyword, remainder = self._split_command(text)

        if kind == "recall_all":
            data = self.memory.recall()
            if not data:
                return "No memories stored."
            lines = ["===== AIVCO MEMORY ====="]
            lines += [f"{k} = {v}" for k, v in data.items()]
            lines.append("=========================")
            return "\n".join(lines)

        if kind == "recall_one":
            key = remainder.split()[0]
            if not key:
                return "Usage: recall <key>"
            if not self.memory.is_valid_key(key):
                return "Invalid key. Use only letters, numbers, '_' or '-' (max 64 characters)."
            value = self.memory.recall(key)
            if value is None:
                return f"No memory found for: {key}"
            return f"{key} = {value}"

        if kind == "recall_bad":
            return "Usage: recall <key>"

        if kind == "forget_bad":
            if text.lower() == "forget":
                return "Usage: forget <key>   (or 'forget all' to clear everything)"
            return "Usage: forget <key>"

        if kind == "forget_all":
            answer = self._confirm_input(
                "This will erase ALL memory. Type 'yes' to confirm: "
            ).strip().lower()
            if answer == "yes":
                self.memory.forget_all()
                return "All memory erased."
            return "Cancelled. No memory was deleted."

        if kind == "forget_one":
            key = remainder.split()[0]
            if not key:
                return "Usage: forget <key>"
            _, msg = self.memory.forget(key)
            return msg

        if kind == "remember_bad":
            return "Usage: remember <key> = <value>"

        if kind == "remember":
            key, value = remainder.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                return "Usage: remember <key> = <value>"
            if not self.memory.is_valid_key(key):
                return "Invalid key. Use only letters, numbers, '_' or '-' (max 64 characters)."
            _, msg = self.memory.remember(key, value)
            return msg

        return None