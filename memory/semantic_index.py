"""Lightweight hybrid semantic index; JSON memory files remain authoritative."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable

import httpx


_WORD_RE = re.compile(r"[a-z0-9]+", re.I)
_EMBED_FAILURE_UNTIL: dict[str, float] = {}
_EMBED_FAILURE_LOCK = threading.Lock()
_SYNONYM_GROUPS = (
    ("car", "automobile", "vehicle", "auto"),
    ("buy", "purchase", "acquire", "shop"),
    ("job", "career", "work", "occupation", "profession"),
    ("happy", "joyful", "glad", "pleased"),
    ("sad", "unhappy", "down", "depressed"),
    ("quick", "fast", "rapid", "speedy"),
    ("home", "house", "residence", "living"),
    ("student", "learner", "pupil", "study"),
    ("doctor", "physician", "medic"),
    ("money", "budget", "finance", "financial", "cost"),
    ("build", "create", "develop", "make", "construct"),
    ("remember", "recall", "memory", "memorize"),
)
_SYNONYMS = {word: group for group in _SYNONYM_GROUPS for word in group}


def _hashed_index(feature: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "big")
    return number % dimensions, (1.0 if number & (1 << 63) else -1.0)


def fallback_embedding(text: str, dimensions: int = 384) -> list[float]:
    """Deterministic signed feature hashing over words and character n-grams."""
    vector = [0.0] * dimensions
    words = _WORD_RE.findall((text or "").lower())
    expanded = list(words)
    for word in words:
        expanded.extend(_SYNONYMS.get(word, ()))
    features = [f"w:{word}" for word in expanded]
    features.extend(f"b:{a}_{b}" for a, b in zip(expanded, expanded[1:]))
    for word in words:
        padded = f"^{word}$"
        features.extend(f"c:{padded[i:i+n]}" for n in (2, 3, 4) for i in range(max(0, len(padded) - n + 1)))
    for feature in features:
        idx, sign = _hashed_index(feature, dimensions)
        vector[idx] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return max(0.0, sum(a * b for a, b in zip(left, right)))


class SemanticIndex:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir or os.environ.get("AIVCO_DATA_DIR", "./data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "semantic_index.db"
        self.embedding_model = os.environ.get("AIVCO_EMBED_MODEL", "nomic-embed-text")
        self.base_url = os.environ.get("AIVCO_LOCAL_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
        self.local_enabled = os.environ.get("AIVCO_LOCAL_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=3.0)
        connection.execute("PRAGMA busy_timeout=3000")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("""CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_mode TEXT NOT NULL
                )""")
                db.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
        except sqlite3.DatabaseError:
            self._remove_database_files()
            with self._connect() as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("""CREATE TABLE memories (
                    memory_id TEXT PRIMARY KEY, category TEXT NOT NULL,
                    content_hash TEXT NOT NULL, updated_at TEXT NOT NULL,
                    vector_json TEXT NOT NULL, content TEXT NOT NULL,
                    embedding_mode TEXT NOT NULL
                )""")
                db.execute("CREATE INDEX idx_memories_category ON memories(category)")

    def _remove_database_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(self.db_path) + suffix).unlink()
            except FileNotFoundError:
                pass

    def _local_embedding(self, text: str) -> list[float] | None:
        with _EMBED_FAILURE_LOCK:
            unavailable_until = _EMBED_FAILURE_UNTIL.get(self.base_url, 0.0)
        if not self.local_enabled or time.monotonic() < unavailable_until:
            return None
        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": self.embedding_model, "input": text},
                timeout=httpx.Timeout(4.0, connect=0.4),
            )
            response.raise_for_status()
            vector = response.json()["data"][0]["embedding"]
            if not isinstance(vector, list) or not vector or not all(isinstance(v, (int, float)) for v in vector):
                return None
            norm = math.sqrt(sum(float(v) ** 2 for v in vector))
            return [float(v) / norm for v in vector] if norm else None
        except (httpx.HTTPError, OSError, ValueError, KeyError, IndexError, TypeError):
            with _EMBED_FAILURE_LOCK:
                _EMBED_FAILURE_UNTIL[self.base_url] = time.monotonic() + 2.0
            return None

    def embed(self, text: str) -> tuple[list[float], str]:
        vector = self._local_embedding(text)
        if vector is not None:
            return vector, f"local:{self.embedding_model}"
        return fallback_embedding(text), "fallback:v1"

    def upsert(self, memory_id: str, category: str, content: str, updated_at: str) -> None:
        content = str(content or "").strip()
        digest = hashlib.sha256(f"{category}\0{content}".encode("utf-8")).hexdigest()
        with self._connect() as db:
            existing = db.execute(
                "SELECT content_hash FROM memories WHERE memory_id=?", (memory_id,)
            ).fetchone()
            if existing and existing[0] == digest:
                db.execute("UPDATE memories SET updated_at=? WHERE memory_id=?", (updated_at, memory_id))
                return
        vector, mode = self.embed(f"{category}: {content}")
        with self._connect() as db:
            db.execute(
                """INSERT INTO memories(memory_id, category, content_hash, updated_at, vector_json, content, embedding_mode)
                   VALUES(?,?,?,?,?,?,?) ON CONFLICT(memory_id) DO UPDATE SET
                   category=excluded.category, content_hash=excluded.content_hash,
                   updated_at=excluded.updated_at, vector_json=excluded.vector_json,
                   content=excluded.content, embedding_mode=excluded.embedding_mode""",
                (memory_id, category, digest, updated_at, json.dumps(vector), content, mode),
            )

    def sync_source(self, source_prefix: str, entries: Iterable[tuple[str, str, str, str]]) -> None:
        current = set()
        for memory_id, category, content, updated_at in entries:
            full_id = f"{source_prefix}{memory_id}"
            current.add(full_id)
            self.upsert(full_id, category, content, updated_at)
        with self._connect() as db:
            existing = db.execute(
                "SELECT memory_id FROM memories WHERE memory_id LIKE ?", (f"{source_prefix}%",)
            ).fetchall()
            stale = [row[0] for row in existing if row[0] not in current]
            db.executemany("DELETE FROM memories WHERE memory_id=?", [(item,) for item in stale])

    def delete(self, memory_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))

    @staticmethod
    def _lexical(query: str, content: str) -> float:
        q_words = set(_WORD_RE.findall(query.lower()))
        c_words = set(_WORD_RE.findall(content.lower()))
        if not q_words:
            return 0.0
        expanded = set(q_words)
        for word in q_words:
            expanded.update(_SYNONYMS.get(word, ()))
        return len(expanded & c_words) / math.sqrt(max(1, len(expanded) * len(c_words)))

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        local_query = self._local_embedding(query)
        fallback_query = fallback_embedding(query)
        with self._connect() as db:
            rows = db.execute(
                "SELECT memory_id, category, updated_at, vector_json, content, embedding_mode FROM memories"
            ).fetchall()
        results = []
        now = __import__("datetime").datetime.now().astimezone()
        for memory_id, category, updated_at, vector_json, content, mode in rows:
            stored = json.loads(vector_json)
            if mode == f"local:{self.embedding_model}" and local_query is not None:
                cosine = _cosine(local_query, stored)
            else:
                cosine = _cosine(fallback_query, fallback_embedding(f"{category}: {content}"))
            lexical = self._lexical(query, f"{category} {content}")
            try:
                dt = __import__("datetime").datetime.fromisoformat(updated_at)
                age_days = max(0.0, (now - dt.astimezone()).total_seconds() / 86400)
            except (TypeError, ValueError):
                age_days = 3650.0
            recency = 1 / (1 + age_days / 30)
            score = 0.62 * cosine + 0.28 * lexical + 0.10 * recency
            if cosine >= 0.08 or lexical >= 0.05:
                results.append({"memory_id": memory_id, "category": category, "content": content, "updated_at": updated_at, "score": score})
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit] if limit is not None else results
