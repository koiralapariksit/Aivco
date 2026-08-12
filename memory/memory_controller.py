"""
AIVCO Memory Controller

Owns all persistent user memory. Talks to a single JSON file on disk
and nothing else -- no subprocess calls, no LLM calls, no model
awareness. This means the memory system keeps working unchanged no
matter what model (Qwen3-0.6B today, something much bigger later)
sits behind AIVCO.
"""

import datetime
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Any

# Keys may only contain letters, numbers, underscore, hyphen.
# This blocks things like "../../etc/passwd" from ever being treated
# as a key, and keeps stored data predictable.
KEY_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
SCHEMA_VERSION = "2.0"
VALID_CATEGORIES = {
    "identity",
    "preference",
    "goal",
    "project",
    "education",
    "career",
    "interest",
    "fact",
    "general",
}
VALID_IMPORTANCE = {"low", "medium", "high", "critical"}
DEFAULT_CATEGORY = "general"
DEFAULT_IMPORTANCE = "medium"

PROFILE_MIGRATION_VERSION = "v1"

UNIFIED_PROFILE_MEMORIES = {
    "name": "Parikshit",
    "preferred_name": "Parikshit",
    "location": "Kathmandu, Nepal",
    "communication_style": "Direct, structured, technically accurate, concise, and action-oriented.",
    "technical_stack": "Python, Dart, Flutter, Django, Kivy/KivyMD, Git, GitHub, VS Code, and local LLM tooling.",
    "education": "NEB Grade 12 science student in Nepal; completed board examinations with a GPA of 3.55.",
    "career_direction": "Primary interest is Data Science and AI engineering; also interested in diplomacy, policy, and foreign service.",
    "study_preferences": "Prefers step-by-step explanations, practical projects, clear academic strategy, and concrete next actions.",
    "goals": "Build AIVCO as a personal local AI assistant and second brain; pursue strong technical and university applications; explore affordable study and scholarship options.",
    "projects": "Actively developing AIVCO and PressureOS; AIVCO is a local personal AI assistant and second brain, and PressureOS is an execution-focused software project.",
    "interests": "FC Barcelona, songwriting, AI/ML, Data Science, diplomacy, and practical software building.",
    "constraints": "Budget for foreign study is approximately NPR 500,000 per year, so scholarships and affordability matter.",
    "uncertainties": "Career direction is still exploratory, but Data Science and AI engineering remain the clearest current priorities.",
}


class MemoryController:
    def __init__(self, memory_path: Path):
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()
        self._had_existing_data = bool(self._data)
        self._apply_profile_migration()

    # ---------------- persistence ----------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.datetime.now().replace(microsecond=0).isoformat()

    @staticmethod
    def _is_record(value: Any) -> bool:
        return (
            isinstance(value, dict)
            and "value" in value
            and "category" in value
            and "importance" in value
            and "created_at" in value
            and "updated_at" in value
        )

    @staticmethod
    def _normalize_category(category: str) -> str:
        return category.strip().lower()

    @staticmethod
    def _normalize_importance(importance: str) -> str:
        return importance.strip().lower()

    @staticmethod
    def is_valid_category(category: str) -> bool:
        return category in VALID_CATEGORIES

    @staticmethod
    def is_valid_importance(importance: str) -> bool:
        return importance in VALID_IMPORTANCE

    def _build_record(
        self,
        value: Any,
        category: str = DEFAULT_CATEGORY,
        importance: str = DEFAULT_IMPORTANCE,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> dict:
        if isinstance(value, dict) and self._is_record(value):
            normalized_category = self._normalize_category(value.get("category", DEFAULT_CATEGORY))
            normalized_importance = self._normalize_importance(value.get("importance", DEFAULT_IMPORTANCE))
            return {
                "value": str(value.get("value", "")).strip(),
                "category": normalized_category if normalized_category in VALID_CATEGORIES else DEFAULT_CATEGORY,
                "importance": normalized_importance if normalized_importance in VALID_IMPORTANCE else DEFAULT_IMPORTANCE,
                "created_at": value.get("created_at") or self._now_iso(),
                "updated_at": value.get("updated_at") or value.get("created_at") or self._now_iso(),
            }
        return {
            "value": str(value).strip(),
            "category": category,
            "importance": importance,
            "created_at": created_at or self._now_iso(),
            "updated_at": updated_at or created_at or self._now_iso(),
        }

    def _extract_records(self, data: dict) -> dict:
        records = {}
        for key, value in data.items():
            if key in {"schema_version", "profile_migration_version"}:
                continue
            if isinstance(value, dict) and self._is_record(value):
                records[self._normalize_key(key)] = self._build_record(value)
            else:
                records[self._normalize_key(key)] = self._build_record(
                    value,
                    category=DEFAULT_CATEGORY,
                    importance=DEFAULT_IMPORTANCE,
                )
        return records

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

    def _apply_profile_migration(self) -> None:
        if (
            self._data.get("schema_version") == SCHEMA_VERSION
            and isinstance(self._data.get("memories"), dict)
        ):
            return

        backup = self.memory_path.with_suffix(".json.bak")
        if self.memory_path.exists() and not backup.exists() and self.memory_path.stat().st_size > 0:
            shutil.copy2(self.memory_path, backup)

        migrated = {
            "schema_version": SCHEMA_VERSION,
            "memories": self._extract_records(self._data),
        }

        if self._had_existing_data:
            for key, value in UNIFIED_PROFILE_MEMORIES.items():
                normalized_key = self._normalize_key(key)
                if normalized_key not in migrated["memories"]:
                    migrated["memories"][normalized_key] = self._build_record(
                        value,
                        category=DEFAULT_CATEGORY,
                        importance=DEFAULT_IMPORTANCE,
                    )

        if migrated != self._data:
            self._data = migrated
            self._save()

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

    def remember(
        self,
        key: str,
        value: str,
        category: str = DEFAULT_CATEGORY,
        importance: str = DEFAULT_IMPORTANCE,
    ) -> tuple[bool, str]:
        key = self._normalize_key(key)
        value = str(value).strip()
        category = self._normalize_category(category)
        importance = self._normalize_importance(importance)

        if not key or not value:
            return False, "Usage: remember <key> = <value>"
        if not self.is_valid_key(key):
            return False, (
                "Invalid key. Use only letters, numbers, '_' or '-' "
                "(max 64 characters)."
            )
        if not self.is_valid_category(category):
            return False, (
                "Invalid category. Use one of: "
                + ", ".join(sorted(VALID_CATEGORIES))
            )
        if not self.is_valid_importance(importance):
            return False, (
                "Invalid importance. Use one of: "
                + ", ".join(sorted(VALID_IMPORTANCE))
            )

        memories = self._data.setdefault("memories", {})
        existing = memories.get(key)
        if existing is not None and self._is_record(existing):
            created_at = existing["created_at"]
        else:
            created_at = None

        memories[key] = self._build_record(
            value,
            category=category,
            importance=importance,
            created_at=created_at,
            updated_at=self._now_iso(),
        )
        self._save()
        return True, f"Saved: {key} = {value}"

    def recall(self, key: Optional[str] = None):
        """Return one value (or None), or the whole key/value dict if key is None."""
        memories = self._data.get("memories", {})
        if key is None:
            return {k: v["value"] for k, v in memories.items()}

        key = self._normalize_key(key)
        if not key:
            return None
        record = memories.get(key)
        if record is None:
            return None
        return record["value"]

    def forget(self, key: str) -> tuple[bool, str]:
        key = self._normalize_key(key)
        if not key:
            return False, "Usage: forget <key>"
        memories = self._data.setdefault("memories", {})
        if key in memories:
            del memories[key]
            self._save()
            return True, f"Forgotten: {key}"
        return False, f"No memory found for: {key}"

    def forget_all(self) -> None:
        self._data = {
            "schema_version": SCHEMA_VERSION,
            "memories": {},
        }
        self._save()

    def is_empty(self) -> bool:
        return len(self._data.get("memories", {})) == 0

    def get_record(self, key: str) -> Optional[dict]:
        key = self._normalize_key(key)
        if not key:
            return None
        return self._data.get("memories", {}).get(key)

    def all_records(self) -> dict:
        return dict(self._data.get("memories", {}))

    def search(self, text: str) -> dict:
        query = str(text).strip().lower()
        if not query:
            return {}
        results = {}
        for key, record in self.all_records().items():
            if (
                query in key.lower()
                or query in record["value"].lower()
                or query in record["category"].lower()
            ):
                results[key] = record
        return results

    def stats(self) -> dict:
        memories = self.all_records()
        total = len(memories)
        importance_counts = {level: 0 for level in VALID_IMPORTANCE}
        category_counts = {category: 0 for category in VALID_CATEGORIES}
        for record in memories.values():
            importance_counts[record["importance"]] = (
                importance_counts.get(record["importance"], 0) + 1
            )
            category_counts[record["category"]] = (
                category_counts.get(record["category"], 0) + 1
            )
        return {
            "total": total,
            "importance": importance_counts,
            "categories": {k: v for k, v in category_counts.items() if v > 0},
        }

    def as_dict(self) -> dict:
        return {k: v["value"] for k, v in self.all_records().items()}