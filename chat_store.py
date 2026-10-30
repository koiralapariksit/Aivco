"""
chat_store.py — AIVCO Multi-Chat Persistence Engine & Project Storage

Handles disk persistence for chats, projects, pins, artifacts, and search.
Enforces strict path traversal security, atomic JSON writes, and corrupt line recovery.
"""

import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
from typing import Optional, Dict, Any, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHATS_DIR = DATA_DIR / "chats"
PROJECTS_DIR = DATA_DIR / "projects"

CHAT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")

INSTRUCTIONS_MAX = 2000
PINS_MAX_ITEMS = 10
PINS_MAX_CHARS = 120
RECAP_MAX = 800
AUTO_TITLE_MAX = 40

_store_lock = threading.Lock()


def sanitize_project_id(project_id: str) -> str:
    """Sanitizes project_id strictly to alphanumeric characters, dashes, and underscores."""
    if not isinstance(project_id, str):
        raise ValueError("Invalid project_id: must be a string")
    if ".." in project_id or "/" in project_id or "\\" in project_id:
        raise ValueError(f"Invalid project_id: path traversal detected in '{project_id}'")
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", project_id).lower().strip("-_")
    if not sanitized:
        raise ValueError(f"Invalid project_id: '{project_id}' contains no valid characters")
    return sanitized


def validate_chat_id(chat_id: str) -> str:
    """Validates that chat_id is strictly a 32-character hexadecimal UUID string."""
    if not isinstance(chat_id, str) or not CHAT_ID_PATTERN.match(chat_id):
        raise ValueError(f"Invalid chat_id '{chat_id}': must match ^[a-f0-9]{{32}}$")
    return chat_id


class ChatStore:
    """Manages chats and project storage with sandboxed paths and thread-safe atomic operations."""

    def __init__(self, chats_dir: Optional[Path] = None, projects_dir: Optional[Path] = None):
        self._chats_dir = (chats_dir or CHATS_DIR).resolve()
        self._projects_dir = (projects_dir or PROJECTS_DIR).resolve()
        self._lock = threading.Lock()

    def get_chats_dir(self) -> Path:
        self._chats_dir.mkdir(parents=True, exist_ok=True)
        return self._chats_dir

    def get_projects_dir(self) -> Path:
        self._projects_dir.mkdir(parents=True, exist_ok=True)
        return self._projects_dir

    def get_chat_dir(self, chat_id: str) -> Path:
        safe_cid = validate_chat_id(chat_id)
        chats_base = self.get_chats_dir()
        target_dir = (chats_base / safe_cid).resolve()
        try:
            target_dir.relative_to(chats_base)
        except ValueError:
            raise ValueError(f"Path traversal detected for chat_id: {chat_id}")
        return target_dir

    def get_project_dir(self, project_id: str) -> Path:
        safe_pid = sanitize_project_id(project_id)
        proj_base = self.get_projects_dir()
        target_dir = (proj_base / safe_pid).resolve()
        try:
            target_dir.relative_to(proj_base)
        except ValueError:
            raise ValueError(f"Path traversal detected for project_id: {project_id}")
        return target_dir

    def _atomic_write_json(self, target_path: Path, data: Any) -> None:
        """Writes JSON atomically via temp file + os.replace."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_name(f".tmp_{target_path.name}_{uuid.uuid4().hex}")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, target_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _atomic_write_text(self, target_path: Path, text: str) -> None:
        """Writes text atomically via temp file + os.replace."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_name(f".tmp_{target_path.name}_{uuid.uuid4().hex}")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, target_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    # ── Chat CRUD ────────────────────────────────────────────────────────────

    def create_chat(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Creates a new persistent chat with server-generated UUID hex."""
        safe_pid = sanitize_project_id(project_id) if project_id else None
        chat_id = uuid.uuid4().hex
        chat_dir = self.get_chat_dir(chat_id)
        chat_dir.mkdir(parents=True, exist_ok=True)

        now = time.time()
        meta = {
            "id": chat_id,
            "title": "New Chat",
            "created": now,
            "updated": now,
            "project_id": safe_pid,
            "starred": False,
            "title_is_manual": False,
            "turn_count": 0,
        }
        self._atomic_write_json(chat_dir / "meta.json", meta)
        self._atomic_write_json(chat_dir / "recap.json", {"lines": [], "covered_through_turn": 0})
        (chat_dir / "messages.jsonl").touch(exist_ok=True)
        return meta

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Returns metadata for a specific chat or None if not found."""
        try:
            chat_dir = self.get_chat_dir(chat_id)
            meta_file = chat_dir / "meta.json"
            if not meta_file.exists():
                return None
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError, OSError):
            return None

    def list_chats(self, project_id: Optional[str] = None, q: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lists all chat metadata, sorted by updated timestamp descending.
        Optionally filters by project_id and/or title substring q.
        """
        chats_base = self.get_chats_dir()
        results: List[Dict[str, Any]] = []
        safe_pid = sanitize_project_id(project_id) if project_id else None
        q_clean = q.strip().lower() if q else None

        with self._lock:
            for item in chats_base.iterdir():
                if not item.is_dir() or not CHAT_ID_PATTERN.match(item.name):
                    continue
                meta_file = item / "meta.json"
                if not meta_file.exists():
                    continue
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if safe_pid is not None and meta.get("project_id") != safe_pid:
                        continue
                    if q_clean and q_clean not in (meta.get("title") or "").lower():
                        continue
                    results.append({
                        "id": meta.get("id", item.name),
                        "title": meta.get("title", "New Chat"),
                        "created": meta.get("created", 0.0),
                        "updated": meta.get("updated", 0.0),
                        "project_id": meta.get("project_id"),
                        "starred": bool(meta.get("starred", False)),
                        "turn_count": meta.get("turn_count", 0),
                    })
                except Exception:
                    continue

        results.sort(key=lambda x: x.get("updated", 0.0), reverse=True)
        return results

    def load_messages(self, chat_id: str, limit: int = 50, before_turn: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Loads messages for a chat in chronological order.
        Gracefully skips corrupt JSON lines.
        """
        chat_dir = self.get_chat_dir(chat_id)
        msg_file = chat_dir / "messages.jsonl"
        if not msg_file.exists():
            return []

        all_records: List[Dict[str, Any]] = []
        with open(msg_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if before_turn is not None and record.get("turn_id", 0) >= before_turn:
                        continue
                    all_records.append(record)
                except Exception:
                    continue  # Gracefully skip corrupt lines

        if limit is not None and limit > 0:
            return all_records[-limit:]
        return all_records

    def append_turn(self, chat_id: str, turn_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Appends a turn record or context_reset to messages.jsonl under lock.
        Performs deterministic auto-titling on the first user prompt if title_is_manual is False.
        """
        chat_dir = self.get_chat_dir(chat_id)
        chat_dir.mkdir(parents=True, exist_ok=True)
        msg_file = chat_dir / "messages.jsonl"

        record = dict(turn_dict)
        if "ts" not in record:
            record["ts"] = time.time()

        with self._lock:
            with open(msg_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Update meta.json updated time, turn_count, and check for auto-titling
            meta = self.get_chat(chat_id)
            if meta:
                meta["updated"] = record["ts"]
                if record.get("turn_id"):
                    meta["turn_count"] = max(meta.get("turn_count", 0), int(record["turn_id"]))
                # Auto-title on the first turn if title has not been manually edited
                if not meta.get("title_is_manual") and record.get("prompt") and record.get("turn_id") == 1:
                    raw_prompt = record["prompt"]
                    # 1. Strip [Attached File...] blocks
                    clean = re.sub(r"\[Attached File:[\s\S]*?```[\s\S]*?```\n?", "", raw_prompt)
                    # 2. Collapse whitespace
                    clean = re.sub(r"\s+", " ", clean).strip()
                    # 3. Cut to 40 chars at word boundary and append '…'
                    if len(clean) > AUTO_TITLE_MAX:
                        cut = clean[:AUTO_TITLE_MAX].rsplit(" ", 1)[0]
                        auto_title = (cut if cut else clean[:AUTO_TITLE_MAX]) + "…"
                    else:
                        auto_title = clean or "New Chat"

                    meta["title"] = auto_title
                self._atomic_write_json(chat_dir / "meta.json", meta)

        return record

    def rename_chat(self, chat_id: str, new_title: str) -> Dict[str, Any]:
        """Renames chat and locks manual title flag."""
        chat_dir = self.get_chat_dir(chat_id)
        clean_title = (new_title or "").strip() or "Untitled Chat"

        with self._lock:
            meta = self.get_chat(chat_id)
            if not meta:
                raise ValueError(f"Chat '{chat_id}' not found")
            meta["title"] = clean_title
            meta["title_is_manual"] = True
            meta["updated"] = time.time()
            self._atomic_write_json(chat_dir / "meta.json", meta)
            return meta

    def set_star(self, chat_id: str, starred: bool) -> Dict[str, Any]:
        """Stars or unstars a chat."""
        chat_dir = self.get_chat_dir(chat_id)
        with self._lock:
            meta = self.get_chat(chat_id)
            if not meta:
                raise ValueError(f"Chat '{chat_id}' not found")
            meta["starred"] = bool(starred)
            meta["updated"] = time.time()
            self._atomic_write_json(chat_dir / "meta.json", meta)
            return meta

    def delete_chat(self, chat_id: str) -> bool:
        """Deletes chat directory from disk."""
        chat_dir = self.get_chat_dir(chat_id)
        with self._lock:
            if not chat_dir.exists():
                return False
            shutil.rmtree(chat_dir, ignore_errors=True)
            return True

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, q: str, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Case-insensitive substring search over chat titles and messages.
        Returns up to 20 newest results with ~100-character snippets.
        """
        q_clean = (q or "").strip().lower()
        if not q_clean:
            return []

        safe_pid = sanitize_project_id(project_id) if project_id else None
        chats_base = self.get_chats_dir()
        results: List[Dict[str, Any]] = []

        with self._lock:
            for item in chats_base.iterdir():
                if not item.is_dir() or not CHAT_ID_PATTERN.match(item.name):
                    continue
                meta_file = item / "meta.json"
                if not meta_file.exists():
                    continue
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if safe_pid is not None and meta.get("project_id") != safe_pid:
                        continue

                    chat_id = meta.get("id", item.name)
                    title = meta.get("title", "")
                    updated = meta.get("updated", 0.0)

                    # 1. Match title
                    if q_clean in title.lower():
                        idx = title.lower().find(q_clean)
                        start = max(0, idx - 40)
                        end = min(len(title), idx + len(q_clean) + 40)
                        snippet = ("..." if start > 0 else "") + title[start:end] + ("..." if end < len(title) else "")
                        results.append({
                            "chat_id": chat_id,
                            "title": title,
                            "project_id": meta.get("project_id"),
                            "updated": updated,
                            "match_type": "title",
                            "snippet": snippet,
                            "turn_id": None,
                        })
                        continue

                    # 2. Match messages
                    msg_file = item / "messages.jsonl"
                    if msg_file.exists():
                        matched_in_chat = False
                        with open(msg_file, "r", encoding="utf-8") as f:
                            for line in f:
                                if matched_in_chat:
                                    break
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    rec = json.loads(line)
                                    text = f"{rec.get('prompt', '')} {rec.get('response', '')}"
                                    if q_clean in text.lower():
                                        idx = text.lower().find(q_clean)
                                        start = max(0, idx - 40)
                                        end = min(len(text), idx + len(q_clean) + 40)
                                        snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
                                        results.append({
                                            "chat_id": chat_id,
                                            "title": title,
                                            "project_id": meta.get("project_id"),
                                            "updated": updated,
                                            "match_type": "message",
                                            "snippet": snippet,
                                            "turn_id": rec.get("turn_id"),
                                        })
                                        matched_in_chat = True
                                except Exception:
                                    continue
                except Exception:
                    continue

        results.sort(key=lambda x: x.get("updated", 0.0), reverse=True)
        return results[:20]

    # ── Project Storage & Pins ───────────────────────────────────────────────

    def create_project(self, project_id: str, instructions: str = "") -> Dict[str, Any]:
        """Creates project directory, instructions.txt, pinned.json, and files/ subdir."""
        safe_pid = sanitize_project_id(project_id)
        proj_dir = self.get_project_dir(safe_pid)
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "files").mkdir(exist_ok=True)

        clean_instr = (instructions or "").strip()[:INSTRUCTIONS_MAX]
        instr_file = proj_dir / "instructions.txt"
        if not instr_file.exists() or clean_instr:
            self._atomic_write_text(instr_file, clean_instr)

        pinned_file = proj_dir / "pinned.json"
        if not pinned_file.exists():
            self._atomic_write_json(pinned_file, [])

        return {
            "project_id": safe_pid,
            "instructions": clean_instr,
            "path": str(proj_dir),
        }

    def read_instructions(self, project_id: str) -> str:
        """Reads project instructions string."""
        proj_dir = self.get_project_dir(project_id)
        instr_file = proj_dir / "instructions.txt"
        if not instr_file.exists():
            return ""
        try:
            return instr_file.read_text(encoding="utf-8").strip()[:INSTRUCTIONS_MAX]
        except Exception:
            return ""

    def write_instructions(self, project_id: str, text: str) -> str:
        """Writes project instructions string atomically."""
        proj_dir = self.get_project_dir(project_id)
        proj_dir.mkdir(parents=True, exist_ok=True)
        clean_text = (text or "").strip()[:INSTRUCTIONS_MAX]
        self._atomic_write_text(proj_dir / "instructions.txt", clean_text)
        return clean_text

    def list_pins(self, project_id: str) -> List[str]:
        """Lists pinned decisions for a project."""
        try:
            proj_dir = self.get_project_dir(project_id)
            pinned_file = proj_dir / "pinned.json"
            if not pinned_file.exists():
                return []
            data = json.loads(pinned_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            return []
        return []

    def add_pin(self, project_id: str, text: str) -> List[str]:
        """Adds a pinned decision (max 10 pins, 120 chars each)."""
        clean_text = (text or "").strip()
        if not clean_text:
            return self.list_pins(project_id)
        clean_text = clean_text[:PINS_MAX_CHARS].strip()

        proj_dir = self.get_project_dir(project_id)
        proj_dir.mkdir(parents=True, exist_ok=True)
        pinned_file = proj_dir / "pinned.json"

        with self._lock:
            pins = self.list_pins(project_id)
            if clean_text in pins:
                return pins
            if len(pins) >= PINS_MAX_ITEMS:
                return pins
            pins.append(clean_text)
            self._atomic_write_json(pinned_file, pins)
            return pins

    def remove_pin(self, project_id: str, index: int) -> List[str]:
        """Removes a pinned decision by index."""
        if not isinstance(index, int):
            return self.list_pins(project_id)

        proj_dir = self.get_project_dir(project_id)
        pinned_file = proj_dir / "pinned.json"

        with self._lock:
            pins = self.list_pins(project_id)
            if 0 <= index < len(pins):
                pins.pop(index)
                self._atomic_write_json(pinned_file, pins)
            return pins

    # ── Artifacts & Recaps ───────────────────────────────────────────────────

    def save_artifact(self, chat_id: str, html: str, note: str = "") -> None:
        """Saves last_app.html and last_app_note.txt atomically in the chat directory."""
        chat_dir = self.get_chat_dir(chat_id)
        chat_dir.mkdir(parents=True, exist_ok=True)
        if html:
            self._atomic_write_text(chat_dir / "last_app.html", html)
        if note:
            self._atomic_write_text(chat_dir / "last_app_note.txt", note.strip() + "\n")

    def load_artifact(self, chat_id: str) -> Optional[Tuple[str, str]]:
        """Loads last_app.html and last_app_note.txt if present."""
        try:
            chat_dir = self.get_chat_dir(chat_id)
            html_file = chat_dir / "last_app.html"
            note_file = chat_dir / "last_app_note.txt"
            if not html_file.exists():
                return None
            html = html_file.read_text(encoding="utf-8").strip()
            note = note_file.read_text(encoding="utf-8").strip() if note_file.exists() else ""
            return html, note
        except Exception:
            return None

    def read_recap(self, chat_id: str) -> Dict[str, Any]:
        """Reads recap.json for a chat."""
        try:
            chat_dir = self.get_chat_dir(chat_id)
            recap_file = chat_dir / "recap.json"
            if not recap_file.exists():
                return {"lines": [], "covered_through_turn": 0}
            data = json.loads(recap_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                return {"lines": data, "covered_through_turn": len(data)}
        except Exception:
            pass
        return {"lines": [], "covered_through_turn": 0}

    def write_recap(self, chat_id: str, lines: List[str], covered_through_turn: int) -> None:
        """Writes recap.json for a chat atomically."""
        chat_dir = self.get_chat_dir(chat_id)
        chat_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "lines": lines,
            "covered_through_turn": covered_through_turn,
        }
        self._atomic_write_json(chat_dir / "recap.json", payload)


# Singleton chat store instance
chat_store = ChatStore()

