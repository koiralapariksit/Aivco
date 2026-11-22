from dataclasses import dataclass
import re
from typing import List
from typing import Any


@dataclass
class MemoryCandidate:
    key: str
    value: str
    category: str
    importance: str


class MemoryDetector:
    """Rule-based, conservative memory detector for AIVCO V1.

    Detects explicit, clear user statements like "My name is X" or
    "I am learning X" and returns MemoryCandidate items for approval.
    """

    # Patterns are ordered; each entry is (regex, category, importance, key_prefix)
    _PATTERNS = [
        # Identity
        (re.compile(r"^(?:my name is|i am called|you can call me)\s+(?P<name>.+)$", re.I), "identity", "high", "name"),
        # Preferences / interests
        (re.compile(r"^(?:i prefer|i like|i don't like|i do not like)\s+(?P<pref>.+)$", re.I), "preference", "medium", "preference"),
        (re.compile(r"^(?:my favorite|my favourites?)\s+(?:is|are)\s+(?P<fav>.+)$", re.I), "interest", "medium", "fav"),
        # Goals
        (re.compile(r"^(?:my goal is|i want to become|i want to be|i plan to become|i plan to be)\s+(?P<goal>.+)$", re.I), "goal", "high", "goal"),
        (re.compile(r"^(?:i want to|i plan to|i am planning to)\s+(?P<plan>.+)$", re.I), "goal", "high", "goal"),
        # Education
        (re.compile(r"^(?:i study|i am studying|i am learning|i am taking)\s+(?P<study>.+)$", re.I), "education", "medium", "learning"),
        # Projects
        (re.compile(r"^(?:i am building|i am working on|my project is)\s+(?P<proj>.+)$", re.I), "project", "high", "project"),
        # Career intent
        (re.compile(r"^(?:i want to work as|i want a career in|i am interested in .* as a career|i am interested in .* as a career)\s*(?P<career>.*)$", re.I), "career", "high", "career"),
    ]

    _CLEAN_KEY_RE = re.compile(r"[^A-Za-z0-9_-]+")

    _TRANSIENT = re.compile(r"\b(weather|tired|hungry|ate|went|today|joke|lol|haha|thanks|thank you)\b", re.I)

    def consolidate_session(self, messages_or_episodes, memory_controller, dry_run: bool = False) -> dict:
        """Extract only explicit durable facts; previews never mutate storage."""
        messages = messages_or_episodes if isinstance(messages_or_episodes, (list, tuple)) else [messages_or_episodes]
        candidates, rejected = [], []
        for item in messages:
            if isinstance(item, dict):
                text = str(item.get("content", item.get("event", ""))).strip()
            else:
                text = str(getattr(item, "event", item)).strip()
            if not text or self._TRANSIENT.search(text):
                rejected.append({"text": text, "reason": "transient_or_empty"})
                continue
            for candidate in self.detect(text):
                existing = memory_controller.search(candidate.value, max_items=5)
                if any(rec["value"].casefold() == candidate.value.casefold() for rec in existing.values()):
                    rejected.append({"text": text, "reason": "duplicate"})
                    break
                candidates.append({"category": candidate.category, "key": candidate.key, "value": candidate.value, "confidence": 0.9, "source": text})
                break
            else:
                rejected.append({"text": text, "reason": "no_durable_fact"})
        accepted = []
        for candidate in candidates:
            if not dry_run:
                memory_controller.upsert_memory(candidate["category"], candidate["key"], candidate["value"], provenance={"source": candidate["source"]})
            accepted.append(candidate)
        return {"accepted": accepted, "rejected": rejected, "dry_run": dry_run}

    def detect(self, user_input: str) -> List[MemoryCandidate]:
        text = user_input.strip()
        if not text:
            return []

        candidates: List[MemoryCandidate] = []

        for pattern, category, importance, prefix in self._PATTERNS:
            m = pattern.match(text)
            if not m:
                continue
            # pick the first named group that matched
            groupdict = m.groupdict()
            value = None
            for v in groupdict.values():
                if v:
                    value = v.strip().rstrip('.')
                    break
            if not value:
                continue

            normalized_value = self._normalize_value(value, category)
            if not normalized_value:
                continue

            raw_key = f"{prefix}_{normalized_value}"
            key = self._clean_key(raw_key)
            if not key:
                continue

            # Normalize value (capitalize first char)
            value_text = normalized_value[0].upper() + normalized_value[1:]

            candidates.append(MemoryCandidate(key=key, value=value_text, category=category, importance=importance))
            break

        return candidates

    def _normalize_value(self, value: str, category: str) -> str:
        normalized = value.strip().rstrip('.')
        if category == "goal":
            while True:
                new = re.sub(
                    r"^(?:to\s+)?(?:become\s+an\s+|become\s+a\s+|become\s+|be\s+|an\s+|a\s+)",
                    "",
                    normalized,
                    flags=re.I,
                ).strip()
                if new == normalized:
                    break
                normalized = new
        elif category == "preference":
            normalized = re.sub(r"^not\s+", "", normalized, flags=re.I).strip()
        return normalized

    def _clean_key(self, raw: str) -> str:
        # replace non-allowed chars with underscore, collapse underscores, trim
        k = raw.strip().lower()
        k = self._CLEAN_KEY_RE.sub("_", k)
        k = re.sub(r"_+", "_", k)
        k = k.strip("_")
        if len(k) == 0:
            return ""
        if len(k) > 64:
            k = k[:64]
        return k

    def _normalize_value(self, value: str, category: str) -> str:
        normalized = value.strip().rstrip('.')
        if category == "goal":
            # Strip grammatical prefixes from goal phrases
            while True:
                new = re.sub(
                    r"^(?:to\s+)?(?:become\s+an\s+|become\s+a\s+|become\s+|be\s+|an\s+|a\s+)",
                    "",
                    normalized,
                    flags=re.I,
                ).strip()
                if new == normalized:
                    break
                normalized = new
        elif category == "preference":
            normalized = re.sub(r"^not\s+", "", normalized, flags=re.I).strip()
        return normalized
