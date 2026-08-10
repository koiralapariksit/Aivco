"""
AIVCO Memory Controller

Owns all persistent user memory. Talks to a single JSON file on disk
and nothing else -- no subprocess calls, no LLM calls, no model
awareness. This means the memory system keeps working unchanged no
matter what model (Qwen3-0.6B today, something much bigger later)
sits behind AIVCO.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

# Keys may only contain letters, numbers, underscore, hyphen.
# This blocks things like "../../etc/passwd" from ever being treated
# as a key, and keeps stored data predictable.
KEY_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class MemoryController:
    def __init__(self, memory_path: Path):
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    # ---------------- persistence ----------------

    def _load(self) -> dict:
        if not self.memory_path.exists():
            self._atomic_write({})
            return {}

        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
        except (json.JSONDecodeError, OSError):
            # Corrupt file: preserve it as a .bak instead of crashing
            # or silently deleting the user's data.
            try:
                backup = self.memory_path.with_suffix(".json.bak")
                self.memory_path.replace(backup)
            except OSError:
                pass
            return {}

    def _atomic_write(self, data: dict) -> None:
        """
        Write to a temp file in the same directory, then atomically
        rename it over the real file. A crash or power loss mid-write
        can never leave profile.json half-written or corrupted.
        """
        directory = self.memory_path.parent
        fd, tmp_path = tempfile.mkstemp(dir=str(directory), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            os.replace(tmp_path, self.memory_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def _save(self) -> None:
        self._atomic_write(self._data)

    # ---------------- validation ----------------

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key.strip().lower()

    @staticmethod
    def is_valid_key(key: str) -> bool:
        return bool(KEY_PATTERN.match(key))

    # ---------------- operations ----------------

    def remember(self, key: str, value: str) -> tuple[bool, str]:
        key = self._normalize_key(key)
        value = value.strip()

        if not key or not value:
            return False, "Usage: remember <key> = <value>"
        if not self.is_valid_key(key):
            return False, (
                "Invalid key. Use only letters, numbers, '_' or '-' "
                "(max 64 characters)."
            )

        self._data[key] = value
        self._save()
        return True, f"Saved: {key} = {value}"

    def recall(self, key: Optional[str] = None):
        """Return one value (or None), or the whole dict if key is None."""
        if key is None:
            return dict(self._data)

        key = self._normalize_key(key)
        if not key:
            return None
        return self._data.get(key)

    def forget(self, key: str) -> tuple[bool, str]:
        key = self._normalize_key(key)
        if not key:
            return False, "Usage: forget <key>"
        if key in self._data:
            del self._data[key]
            self._save()
            return True, f"Forgotten: {key}"
        return False, f"No memory found for: {key}"

    def forget_all(self) -> None:
        self._data = {}
        self._save()

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def as_dict(self) -> dict:
        return dict(self._data)