"""
AIVCO Kernel Cognitive Arbiter, Executive Context Broker & Session Attribution Ledger
Deterministically classifies incoming user queries into Kernel (Zero-LLM),
Brain 4 Web AI (Claude/ChatGPT via Extension Bridge), or Tier 2/3 Fast API (Groq/Gemini).
Runs in <5ms with 0-LLM token consumption.
"""

from collections import deque, OrderedDict
from contextvars import ContextVar
from datetime import datetime, timezone
import difflib
import json
import logging
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("aivco.arbiter")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TELEMETRY_FILE = BASE_DIR / "data" / "arbiter_training_log.jsonl"
PROJECTS_DIR = BASE_DIR / "data" / "projects"


# ── Asynchronous Telemetry Logger Queue & Daemon Worker ───────────────────────

_telemetry_queue: queue.Queue = queue.Queue(maxsize=50000)
_custom_telemetry_path: Optional[Path] = None
_telemetry_error_logged: bool = False


def _telemetry_worker() -> None:
    global _telemetry_error_logged
    while True:
        try:
            item = _telemetry_queue.get(timeout=0.05)
            batch = [item]
            while not _telemetry_queue.empty() and len(batch) < 200:
                try:
                    batch.append(_telemetry_queue.get_nowait())
                except queue.Empty:
                    break

            grouped: Dict[Path, List[Dict[str, Any]]] = {}
            for entry, target_path in batch:
                grouped.setdefault(target_path, []).append(entry)

            for target_path, entries in grouped.items():
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(target_path, "a", encoding="utf-8") as f:
                        for entry in entries:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception as exc:
                    if not _telemetry_error_logged:
                        _telemetry_error_logged = True
                        sys.stderr.write(f"[aivco-arbiter] Telemetry write warning: {exc}\n")

            for _ in batch:
                _telemetry_queue.task_done()
        except queue.Empty:
            continue
        except Exception:
            pass


_worker_thread = threading.Thread(
    target=_telemetry_worker, daemon=True, name="ArbiterTelemetryWorker"
)
_worker_thread.start()


def set_telemetry_filepath(filepath: Optional[str]) -> None:
    """Sets a custom filepath for telemetry logging (useful for unit testing)."""
    global _custom_telemetry_path
    _custom_telemetry_path = Path(filepath) if filepath else None


def flush_telemetry(timeout: float = 2.0) -> None:
    """Drains the telemetry queue up to timeout seconds without hanging indefinitely."""
    deadline = time.perf_counter() + max(0.05, timeout)
    while not _telemetry_queue.empty() and time.perf_counter() < deadline:
        time.sleep(0.01)


def log_arbiter_telemetry(
    prompt: str,
    target_tier: str,
    rationale: str,
    latency_ms: float,
    filepath: Optional[str] = None,
) -> None:
    """Non-blocking asynchronous append to arbiter_training_log.jsonl."""
    target_path = (
        Path(filepath)
        if filepath
        else (_custom_telemetry_path or DEFAULT_TELEMETRY_FILE)
    )
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "target_tier": target_tier,
        "rationale": rationale,
        "latency_ms": latency_ms,
    }
    try:
        _telemetry_queue.put_nowait((entry, target_path))
    except queue.Full:
        try:
            _telemetry_queue.get_nowait()
            _telemetry_queue.task_done()
            _telemetry_queue.put_nowait((entry, target_path))
        except Exception:
            pass


# ── Patterns & Provider Alias Dictionaries ────────────────────────────────────

IDENTITY_PATTERN = re.compile(
    r"^(?:who\s+am\s+i|who\s+the\s+hell\s+am\s+i|tell\s+(?:me\s+)?a?bout\s+me|(?:what\s+do\s+you\s+)?know\s+(?:me\s+|a?bout\s+me)|what\s+do\s+you\s+know\s+about\s+me|my\s+profile|full\s+profile)[?.!]*$",
    re.IGNORECASE,
)

META_CONVERSATION_EXCLUDE = re.compile(
    r"\b(trying\s+to\s+say|summarize|in\s+short|above\s+chat|previous|what\s+did|chatgpt|chatpgt|claude|gpt|gemini|grok)\b",
    re.IGNORECASE,
)

ATTRIBUTION_PATTERN = re.compile(
    r"^(?:which|what)\s+(?:ai\s+)?model\s+(?:did\s+you\s+use|made|built|generated|was\s+used)|for\s+.*?\s+which\s+model\s+did\s+you\s+use|^who\s+(?:built|made|generated)\s+(?:this|that|it)\??$",
    re.IGNORECASE,
)

PIN_SHORTCUT_PATTERN = re.compile(
    r"^(?:remember\s+to\s+pin|pin)\s*:\s*(.+)$",
    re.IGNORECASE,
)

LIVE_SEARCH_PATTERN = re.compile(
    r"\b(today|yesterday|latest|recent|current|news|score|scores|match|matches|game|games|result|results|standing|standings|live|2025|2026|last\s+(?:match|game|night)|next\s+(?:match|game))\b",
    re.IGNORECASE,
)

BACKEND_SCRIPT_GUARD = re.compile(
    r"\b(python|pyhton|tkinter|cli|terminal|bash|powershell|fastapi|django|flask|sql|postgresql|pytorch|pandas|numpy|scikit|c\+\+|rust)\b",
    re.IGNORECASE,
)

FRONTEND_PATTERN = re.compile(
    r"\b(web(?:site|asite)|e-?commer(?:ce|ice)|clon(?:e|ing|ed)|claong|dashboard|html|web\s*app|store|spotify|netflix|nelflix|youtube|ui(?:\s+only)?|interface|landing\s+page|working\s+app|build\s+it|make\s+it|three\.js|canvas)\b",
    re.IGNORECASE,
)

CODING_PATTERN = re.compile(
    r"\b(py(?:thon|hton)|script|scirpt|tkinter|gui|algorithm|api|backend|endpoint|fastapi|function|class|sql|regex|debug|refactor)\b",
    re.IGNORECASE,
)

ITERATIVE_REFINEMENT_PATTERN = re.compile(
    r"^(?:now\s+|please\s+|can\s+you\s+)?(?:add|change|update|modify|make\s+it|fix|remove|replace|put|include|switch\s+the\s+theme|make\s+the)\b",
    re.IGNORECASE,
)

KNOWN_PROVIDER_ALIASES: Dict[str, str] = {
    # Claude (Brain 4 Web)
    "claude": "claude",
    "cladue": "claude",
    "claud": "claude",
    "cloude": "claude",
    "calude": "claude",
    "claudeai": "claude",
    "claude_web": "claude",
    "claudeto": "claude",
    "cladueto": "claude",
    # ChatGPT (Brain 4 Web)
    "chatgpt": "chatgpt",
    "chat gpt": "chatgpt",
    "gpt": "chatgpt",
    "chagpt": "chatgpt",
    "chatgtp": "chatgpt",
    "openai": "chatgpt",
    "gpt4": "chatgpt",
    "gpt4o": "chatgpt",
    "chatgpt_web": "chatgpt",
    "gptto": "chatgpt",
    "chatgptto": "chatgpt",
    # Gemini Cloud API (Tier 3 Fast Cloud API Key)
    "gemini": "gemini",
    "gemeni": "gemini",
    "gmini": "gemini",
    "gemni": "gemini",
    "gemini api": "gemini",
    "gemeni api": "gemini",
    "gmini api": "gemini",
    "gemini key": "gemini",
    "gemeni key": "gemini",
    "gemini pro": "gemini",
    "gemini flash": "gemini",
    "gemeni pro": "gemini",
    "gemeni flash": "gemini",
    "gemini-flash": "gemini",
    "gemini-pro": "gemini",
    # Gemini Web (Browser Account Automation)
    "gemini web": "gemini_web",
    "gemini_web": "gemini_web",
    "gemini browser": "gemini_web",
    "gemeni web": "gemini_web",
    "gemeni browser": "gemini_web",
    # Grok (Brain 4 Web)
    "grok": "grok",
    "gork": "grok",
    "xai": "grok",
    "grok_web": "grok",
    # Groq (Tier 2 Fast Cloud API)
    "groq": "groq",
    "grq": "groq",
    "groq api": "groq",
    "llama": "groq",
    # Local / Kernel
    "local": "local_llama",
    "local_llama": "local_llama",
    "ollama": "local_llama",
    "kernel": "kernel",
    "aivco-kernel": "kernel",
}

CANONICAL_PROVIDERS: Dict[str, List[str]] = {
    "claude": ["claude", "cladue", "cloude"],
    "chatgpt": ["chatgpt", "gpt", "openai"],
    "gemini": ["gemini", "gemeni", "gemini api", "gemeni api", "gemini pro"],
    "gemini_web": ["gemini web", "gemeni web", "gemini browser"],
    "grok": ["grok", "gork", "xai"],
    "groq": ["groq", "groq api"],
    "local_llama": ["local", "ollama"],
}

WEB_PROVIDERS = frozenset(
    {"claude", "chatgpt", "gemini_web", "grok", "claude_web", "chatgpt_web", "grok_web"}
)

HISTORY_EXCHANGES = 16
HISTORY_MAX_CHARS = 30000
REPLY_MAX_CHARS = 1500
PROVIDER_EXCHANGE_OVERRIDE = {"groq": 8}
INSTRUCTIONS_MAX = 2000
PINS_MAX_ITEMS = 10
RECAP_MAX = 800
RAM_LOADED_CHATS = 20

_current_chat_ctx: ContextVar[Optional[Tuple[Optional[str], bool, Optional[str]]]] = ContextVar("current_chat_ctx", default=None)


def set_current_chat(chat_id: Optional[str], temporary: bool = False, project_id: Optional[str] = None) -> None:
    """Binds the current execution context to a specific chat_id, temporary flag, and project_id."""
    _current_chat_ctx.set((chat_id, temporary, project_id))


def get_current_chat() -> Optional[Tuple[Optional[str], bool, Optional[str]]]:
    """Returns the current bound chat context tuple (chat_id, temporary, project_id)."""
    return _current_chat_ctx.get()


# ── Session Attribution & Artifact Ledger ────────────────────────────────────

def sanitize_project_id(project_id: str) -> str:
    """
    Sanitizes project_id strictly to alphanumeric characters, dashes, and underscores.
    Raises ValueError on directory traversal attempts or invalid/empty names.
    """
    if not isinstance(project_id, str):
        raise ValueError("Invalid project_id: must be a string")
    if ".." in project_id or "/" in project_id or "\\" in project_id:
        raise ValueError(f"Invalid project_id: path traversal or separator detected in '{project_id}'")
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", project_id).lower().strip("-_")
    if not sanitized:
        raise ValueError(f"Invalid project_id: '{project_id}' contains no valid characters")
    return sanitized


def ensure_project_storage(
    project_id: str,
    instructions: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """
    Initializes and validates the persistent on-disk storage directory for a project:
      data/projects/<project_id>/
      data/projects/<project_id>/files/
      data/projects/<project_id>/instructions.txt (if supplied)
      data/projects/<project_id>/chat_history.jsonl
      data/projects/<project_id>/last_app.html
      data/projects/<project_id>/pinned.json
      data/projects/<project_id>/recap.json
    """
    safe_pid = sanitize_project_id(project_id)
    target_base = (base_dir or PROJECTS_DIR).resolve()
    target_base.mkdir(parents=True, exist_ok=True)
    root = (target_base / safe_pid).resolve()

    try:
        root.relative_to(target_base)
    except ValueError:
        raise ValueError(f"Path traversal detected: {project_id}")

    root.mkdir(parents=True, exist_ok=True)
    (root / "files").mkdir(exist_ok=True)
    if instructions is not None:
        (root / "instructions.txt").write_text(instructions, encoding="utf-8")
    chat_file = root / "chat_history.jsonl"
    if not chat_file.exists():
        chat_file.touch()
    html_file = root / "last_app.html"
    if not html_file.exists():
        html_file.touch()
    pinned_file = root / "pinned.json"
    if not pinned_file.exists():
        pinned_file.write_text("[]\n", encoding="utf-8")
    recap_file = root / "recap.json"
    if not recap_file.exists():
        recap_file.write_text("[]\n", encoding="utf-8")
    return root


class SessionTurnRecord:
    __slots__ = (
        "turn_id",
        "prompt",
        "target_tier",
        "actual_provider",
        "cascaded_from",
        "has_html_artifact",
        "html_code",
        "response",
        "timestamp",
    )

    def __init__(
        self,
        turn_id: int,
        prompt: str,
        target_tier: str,
        actual_provider: str,
        cascaded_from: Optional[str] = None,
        has_html_artifact: bool = False,
        html_code: str = "",
        response: str = "",
        timestamp: Optional[float] = None,
    ):
        self.turn_id = turn_id
        self.prompt = prompt
        self.target_tier = target_tier
        self.actual_provider = actual_provider
        self.cascaded_from = cascaded_from
        self.has_html_artifact = has_html_artifact
        self.html_code = html_code
        self.response = response
        self.timestamp = timestamp if timestamp is not None else time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "prompt": self.prompt,
            "target_tier": self.target_tier,
            "actual_provider": self.actual_provider,
            "cascaded_from": self.cascaded_from,
            "has_html_artifact": self.has_html_artifact,
            "html_code": self.html_code,
            "response": self.response,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionTurnRecord":
        return cls(
            turn_id=data.get("turn_id", 0),
            prompt=data.get("prompt", ""),
            target_tier=data.get("target_tier", ""),
            actual_provider=data.get("actual_provider", ""),
            cascaded_from=data.get("cascaded_from"),
            has_html_artifact=data.get("has_html_artifact", False),
            html_code=data.get("html_code", ""),
            response=data.get("response", ""),
            timestamp=data.get("timestamp", time.time()),
        )


class SessionExecutionLedger:
    """
    Tracks execution turns, model attribution, and active HTML artifacts.
    Supports two operating modes:
    1. Casual Mode (mode="casual" or missing project_id):
       - Ephemeral in-memory rolling deque(maxlen=12).
       - Never writes to or reads from disk.
    2. Project Mode (mode="project" with valid project_id):
       - Persists all turns to data/projects/<project_id>/chat_history.jsonl (append-only).
       - Persists latest HTML artifact to data/projects/<project_id>/last_app.html.
       - Survives server restarts by restoring turns on demand.
       - Fully isolated across project IDs and from Casual Mode.
    """

    PROVIDER_DISPLAY_NAMES = {
        "claude": "Claude 3.7 Sonnet (via AIVCO Web AI Bridge)",
        "chatgpt": "ChatGPT / GPT-4o (via AIVCO Web AI Bridge)",
        "gemini_web": "Gemini Advanced (via AIVCO Web AI Bridge)",
        "grok": "Grok 3 (via AIVCO Web AI Bridge)",
        "gemini": "Google Gemini 2.5 Pro/Flash (Tier 3 Cloud API)",
        "groq": "Llama 3.3 70B Versatile (via Groq Tier 2 Fast API)",
        "local_llama": "Local Quantized GGUF SLM (Tier 1 Offline Engine)",
        "kernel": "AIVCO Cognitive Kernel (Stage 0 Deterministic Engine)",
    }

    def __init__(self, max_turns: int = 20, projects_dir: Optional[Path] = None):
        self._lock = threading.Lock()
        self._max_turns: int = max_turns
        self._turns: Deque[SessionTurnRecord] = deque(maxlen=max_turns)
        self._casual_turns: List[SessionTurnRecord] = []
        self._casual_recap: List[str] = []
        self._turn_counter: int = 0
        self._project_ledgers: Dict[str, Deque[SessionTurnRecord]] = {}
        self._chat_lru: OrderedDict[str, Deque[SessionTurnRecord]] = OrderedDict()
        self._temporary_turns: Deque[SessionTurnRecord] = deque(maxlen=50)
        self._projects_dir: Path = projects_dir or PROJECTS_DIR

    def set_current_chat(self, chat_id: Optional[str], temporary: bool = False, project_id: Optional[str] = None) -> None:
        set_current_chat(chat_id, temporary=temporary, project_id=project_id)

    def get_current_chat(self) -> Optional[Tuple[Optional[str], bool, Optional[str]]]:
        return get_current_chat()

    def _resolve_active_ctx(
        self,
        mode: str = "casual",
        project_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        temporary: Optional[bool] = None,
    ) -> Tuple[Optional[str], bool, Optional[str]]:
        ctx = _current_chat_ctx.get()
        active_chat_id = chat_id or (ctx[0] if ctx else None)
        active_temp = temporary if temporary is not None else (ctx[1] if ctx else False)
        active_pid = project_id or (ctx[2] if ctx else None)
        return active_chat_id, active_temp, active_pid

    def _get_or_load_chat_turns(self, chat_id: str) -> Deque[SessionTurnRecord]:
        safe_cid = chat_id.strip().lower()
        if safe_cid in self._chat_lru:
            self._chat_lru.move_to_end(safe_cid)
            return self._chat_lru[safe_cid]

        d: Deque[SessionTurnRecord] = deque(maxlen=50)
        try:
            from chat_store import chat_store
            messages = chat_store.load_messages(safe_cid, limit=50)
            for m in messages:
                if m.get("type") == "context_reset":
                    d.append(SessionTurnRecord(turn_id=0, prompt="", target_tier="", actual_provider="", response="[Context Reset]"))
                elif "prompt" in m:
                    d.append(SessionTurnRecord(
                        turn_id=m.get("turn_id", 0),
                        prompt=m.get("prompt", ""),
                        target_tier=m.get("target_tier", "kernel"),
                        actual_provider=m.get("actual_provider") or m.get("provider", "groq"),
                        cascaded_from=m.get("cascaded_from"),
                        has_html_artifact=bool(m.get("has_html", False)),
                        html_code=m.get("html_code", ""),
                        response=m.get("response", ""),
                    ))
        except Exception:
            pass

        if len(self._chat_lru) >= RAM_LOADED_CHATS:
            self._chat_lru.popitem(last=False)
        self._chat_lru[safe_cid] = d
        return d

    def _sync_chat_recap(self, chat_id: str) -> None:
        try:
            from chat_store import chat_store
            safe_cid = chat_id.strip().lower()
            all_msgs = chat_store.load_messages(safe_cid, limit=200)
            reset_idx = -1
            for idx, m in enumerate(all_msgs):
                if m.get("type") == "context_reset":
                    reset_idx = idx
            active_msgs = all_msgs[reset_idx + 1:] if reset_idx >= 0 else all_msgs
            turn_msgs = [m for m in active_msgs if "prompt" in m]
            if len(turn_msgs) <= HISTORY_EXCHANGES:
                chat_store.write_recap(safe_cid, [], covered_through_turn=0)
                return

            evicted = turn_msgs[:-HISTORY_EXCHANGES]
            raw_lines: List[str] = []
            for t in evicted:
                p_text = (t.get("prompt") or "").strip()[:80].strip()
                prov = t.get("provider") or t.get("actual_provider") or "groq"
                app_flag = " [app built]" if t.get("has_html") else ""
                raw_lines.append(f"{p_text} -> {prov}{app_flag}")

            compressed = self._compress_recap_lines(raw_lines, max_chars=RECAP_MAX)
            covered_turn = evicted[-1].get("turn_id", 0)
            chat_store.write_recap(safe_cid, compressed, covered_through_turn=covered_turn)
        except Exception:
            pass

    def _sanitize_project_id(self, project_id: str) -> str:
        return sanitize_project_id(project_id)

    def get_project_dir(self, project_id: str) -> Path:
        safe_pid = sanitize_project_id(project_id)
        target_base = self._projects_dir.resolve()
        target_base.mkdir(parents=True, exist_ok=True)
        proj_dir = (target_base / safe_pid).resolve()
        try:
            proj_dir.relative_to(target_base)
        except ValueError:
            raise ValueError(f"Path traversal detected: {project_id}")
        return proj_dir

    def ensure_project_storage(self, project_id: str, instructions: Optional[str] = None) -> Path:
        return ensure_project_storage(project_id, instructions=instructions, base_dir=self._projects_dir)

    def _is_project_mode(self, mode: str = "casual", project_id: Optional[str] = None) -> bool:
        return (mode == "project" or bool(project_id)) and bool(project_id and project_id.strip())

    def _get_or_load_project_turns(self, project_id: str) -> Deque[SessionTurnRecord]:
        safe_pid = self._sanitize_project_id(project_id)
        if safe_pid in self._project_ledgers:
            return self._project_ledgers[safe_pid]

        d: Deque[SessionTurnRecord] = deque(maxlen=50)
        proj_dir = self.get_project_dir(safe_pid)
        history_file = proj_dir / "chat_history.jsonl"
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            d.append(SessionTurnRecord.from_dict(data))
                        except Exception:
                            continue
            except Exception:
                pass

        self._project_ledgers[safe_pid] = d
        return d

    def _load_all_project_turns(self, project_id: str) -> List[SessionTurnRecord]:
        safe_pid = self._sanitize_project_id(project_id)
        proj_dir = self.get_project_dir(safe_pid)
        history_file = proj_dir / "chat_history.jsonl"
        turns: List[SessionTurnRecord] = []
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            turns.append(SessionTurnRecord.from_dict(json.loads(line)))
                        except Exception:
                            continue
            except Exception:
                pass
        return turns

    def list_pins(self, project_id: str) -> List[str]:
        try:
            safe_pid = self._sanitize_project_id(project_id)
            proj_dir = self.get_project_dir(safe_pid)
        except ValueError:
            return []
        pinned_file = proj_dir / "pinned.json"
        if not pinned_file.exists():
            return []
        try:
            data = json.loads(pinned_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            return []
        return []

    def pin(self, project_id: str, text: str) -> bool:
        if not text or not isinstance(text, str):
            return False
        clean_text = text.strip()
        if not clean_text:
            return False
        clean_text = clean_text[:120].strip()

        safe_pid = self._sanitize_project_id(project_id)
        proj_dir = self.get_project_dir(safe_pid)
        proj_dir.mkdir(parents=True, exist_ok=True)
        pinned_file = proj_dir / "pinned.json"

        pins = self.list_pins(safe_pid)
        if clean_text in pins:
            return True
        if len(pins) >= 10:
            return False

        pins.append(clean_text)
        pinned_file.write_text(json.dumps(pins, indent=2), encoding="utf-8")
        return True

    def unpin(self, project_id: str, index: int) -> bool:
        if not isinstance(index, int):
            return False
        safe_pid = self._sanitize_project_id(project_id)
        proj_dir = self.get_project_dir(safe_pid)
        pinned_file = proj_dir / "pinned.json"

        pins = self.list_pins(safe_pid)
        if 0 <= index < len(pins):
            pins.pop(index)
            pinned_file.write_text(json.dumps(pins, indent=2), encoding="utf-8")
            return True
        return False

    def _compress_recap_lines(self, lines: List[str], max_chars: int = 800) -> List[str]:
        if not lines:
            return []

        def fits(cand: List[str]) -> bool:
            return len(json.dumps(cand)) <= max_chars and len("\n".join(cand)) <= max_chars

        if fits(lines):
            return lines

        for n in range(2, len(lines) + 1):
            milestone = f"[Early turns 1-{n}: initial setup, backend schemas, and UI scaffold]"
            cand = [milestone] + lines[n:]
            if fits(cand):
                return cand

        return [f"[Early turns 1-{len(lines)}: initial setup, backend schemas, and UI scaffold]"]

    def _sync_project_recap(self, project_id: str) -> None:
        safe_pid = self._sanitize_project_id(project_id)
        proj_dir = self.get_project_dir(safe_pid)
        recap_file = proj_dir / "recap.json"

        turns = self._load_all_project_turns(safe_pid)
        t_count = len(turns)
        if t_count <= HISTORY_EXCHANGES:
            recap_file.write_text("[]\n", encoding="utf-8")
            return

        evicted = turns[:-HISTORY_EXCHANGES]
        raw_lines: List[str] = []
        for t in evicted:
            p_text = (t.prompt or "").strip()[:80].strip()
            prov = t.actual_provider or "groq"
            app_flag = " [app built]" if t.has_html_artifact else ""
            raw_lines.append(f"{p_text} -> {prov}{app_flag}")

        compressed = self._compress_recap_lines(raw_lines, max_chars=800)
        recap_file.write_text(json.dumps(compressed, indent=2), encoding="utf-8")

    def get_recap_entries(self, project_id: str) -> List[str]:
        try:
            safe_pid = self._sanitize_project_id(project_id)
            proj_dir = self.get_project_dir(safe_pid)
        except ValueError:
            return []
        recap_file = proj_dir / "recap.json"
        if not recap_file.exists():
            return []
        try:
            data = json.loads(recap_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            return []
        return []

    def _compress_casual_recap_lines(self, lines: List[str], max_chars: int = 800) -> List[str]:
        if not lines:
            return []

        def fits(cand: List[str]) -> bool:
            return len("\n".join(cand)) <= max_chars

        if fits(lines):
            return lines

        for n in range(2, len(lines) + 1):
            milestone = "[Earlier session context: user exploration and Q&A]"
            cand = [milestone] + lines[n:]
            if fits(cand):
                return cand

        return ["[Earlier session context: user exploration and Q&A]"]

    def _sync_casual_recap(self) -> None:
        t_count = len(self._casual_turns)
        if t_count <= HISTORY_EXCHANGES:
            self._casual_recap = []
            return

        evicted = self._casual_turns[:-HISTORY_EXCHANGES]
        raw_lines: List[str] = []
        for t in evicted:
            p_text = (t.prompt or "").strip()[:80].strip()
            prov = t.actual_provider or "groq"
            app_flag = " [app built]" if t.has_html_artifact else ""
            raw_lines.append(f"{p_text} -> {prov}{app_flag}")

        self._casual_recap = self._compress_casual_recap_lines(raw_lines, max_chars=800)

    def get_casual_recap_entries(self) -> List[str]:
        with self._lock:
            return list(self._casual_recap)

    def record_turn(
        self,
        prompt: str,
        target_tier: str,
        actual_provider: Optional[str] = None,
        cascaded_from: Optional[str] = None,
        response_markdown: str = "",
        mode: str = "casual",
        project_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        temporary: Optional[bool] = None,
    ) -> SessionTurnRecord:
        provider = (actual_provider or target_tier or "groq").strip().lower()
        if ":" in provider:
            provider = provider.split(":", 1)[1].strip()
        provider = KNOWN_PROVIDER_ALIASES.get(provider, provider)

        cascaded = (cascaded_from or "").strip().lower()
        if ":" in cascaded:
            cascaded = cascaded.split(":", 1)[1].strip()
        cascaded = KNOWN_PROVIDER_ALIASES.get(cascaded, cascaded) if cascaded else None

        html_code = ""
        has_html = False
        if response_markdown and "<!doctype html" in response_markdown.lower():
            has_html = True
            m = re.search(r"```html\s*([\s\S]*?)```", response_markdown, re.IGNORECASE)
            html_code = m.group(1).strip() if m else response_markdown.strip()

        with self._lock:
            active_chat_id, active_temp, active_pid = self._resolve_active_ctx(
                mode=mode, project_id=project_id, chat_id=chat_id, temporary=temporary
            )

            # 1. Temporary Mode: purely in-memory
            if active_temp:
                self._turn_counter += 1
                rec = SessionTurnRecord(
                    turn_id=self._turn_counter,
                    prompt=prompt,
                    target_tier=target_tier,
                    actual_provider=provider,
                    cascaded_from=cascaded if cascaded != provider else None,
                    has_html_artifact=has_html,
                    html_code=html_code,
                    response=response_markdown,
                )
                self._temporary_turns.append(rec)
                return rec

            # 2. Active Chat Mode: persisted via chat_store
            if active_chat_id:
                from chat_store import chat_store
                safe_cid = active_chat_id.strip().lower()
                d = self._get_or_load_chat_turns(safe_cid)
                highest_turn = max([t.turn_id for t in d if t.turn_id > 0], default=0)
                turn_id = highest_turn + 1
                rec = SessionTurnRecord(
                    turn_id=turn_id,
                    prompt=prompt,
                    target_tier=target_tier,
                    actual_provider=provider,
                    cascaded_from=cascaded if cascaded != provider else None,
                    has_html_artifact=has_html,
                    html_code=html_code,
                    response=response_markdown,
                )
                d.append(rec)
                self._chat_lru.move_to_end(safe_cid)

                turn_payload = {
                    "turn_id": turn_id,
                    "ts": time.time(),
                    "prompt": prompt,
                    "response": response_markdown,
                    "provider": provider,
                    "target_tier": target_tier,
                    "cascaded_from": cascaded if cascaded != provider else None,
                    "has_html": has_html,
                    "html_code": html_code,
                }
                chat_store.append_turn(safe_cid, turn_payload)

                if has_html and html_code:
                    chat_store.save_artifact(safe_cid, html_code, f"App updated on turn {turn_id} by {provider}")

                if active_pid:
                    try:
                        safe_pid = sanitize_project_id(active_pid)
                        m_pin = PIN_SHORTCUT_PATTERN.match(prompt.strip())
                        if m_pin:
                            chat_store.add_pin(safe_pid, m_pin.group(1).strip())
                            try:
                                self.pin(safe_pid, m_pin.group(1).strip())
                            except Exception:
                                pass
                        proj_dir = self.get_project_dir(safe_pid)
                        proj_dir.mkdir(parents=True, exist_ok=True)
                        history_file = proj_dir / "chat_history.jsonl"
                        with open(history_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec.to_dict()) + "\n")
                        if has_html and html_code:
                            (proj_dir / "last_app.html").write_text(html_code, encoding="utf-8")
                            (proj_dir / "last_app_note.txt").write_text(f"App updated on turn {turn_id} by {provider}\n", encoding="utf-8")
                    except Exception:
                        pass

                self._sync_chat_recap(safe_cid)
                return rec

            # 3. Fallback Legacy Mode (when no chat context is set)
            if self._is_project_mode(mode, project_id):
                assert project_id is not None
                safe_pid = self._sanitize_project_id(project_id)
                proj_turns = self._get_or_load_project_turns(safe_pid)
                turn_id = (proj_turns[-1].turn_id + 1) if proj_turns else 1

                rec = SessionTurnRecord(
                    turn_id=turn_id,
                    prompt=prompt,
                    target_tier=target_tier,
                    actual_provider=provider,
                    cascaded_from=cascaded if cascaded != provider else None,
                    has_html_artifact=has_html,
                    html_code=html_code,
                    response=response_markdown,
                )
                proj_turns.append(rec)

                # Persist turn append-only to disk
                proj_dir = self.get_project_dir(safe_pid)
                proj_dir.mkdir(parents=True, exist_ok=True)
                (proj_dir / "files").mkdir(exist_ok=True)
                history_file = proj_dir / "chat_history.jsonl"
                with open(history_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec.to_dict()) + "\n")

                # Natural language pin shortcut
                m_pin = PIN_SHORTCUT_PATTERN.match(prompt.strip())
                if m_pin:
                    pin_text = m_pin.group(1).strip()
                    try:
                        self.pin(safe_pid, pin_text)
                    except Exception:
                        pass

                # If an HTML artifact is present, write directly to last_app.html and update last_app_note.txt
                if has_html and html_code:
                    artifact_file = proj_dir / "last_app.html"
                    artifact_file.write_text(html_code, encoding="utf-8")
                    note_file = proj_dir / "last_app_note.txt"
                    note_file.write_text(f"App updated on turn {rec.turn_id} by {provider}\n", encoding="utf-8")

                # Sync deterministic rolling recap
                self._sync_project_recap(safe_pid)

                return rec
            else:
                # Casual mode: purely in-memory
                self._turn_counter += 1
                rec = SessionTurnRecord(
                    turn_id=self._turn_counter,
                    prompt=prompt,
                    target_tier=target_tier,
                    actual_provider=provider,
                    cascaded_from=cascaded if cascaded != provider else None,
                    has_html_artifact=has_html,
                    html_code=html_code,
                    response=response_markdown,
                )
                self._turns.append(rec)
                self._casual_turns.append(rec)
                self._sync_casual_recap()
                return rec

    def get_last_turn(self, mode: str = "casual", project_id: Optional[str] = None, chat_id: Optional[str] = None) -> Optional[SessionTurnRecord]:
        with self._lock:
            active_chat_id, active_temp, active_pid = self._resolve_active_ctx(mode=mode, project_id=project_id, chat_id=chat_id)
            if active_temp:
                return self._temporary_turns[-1] if self._temporary_turns else None
            if active_chat_id:
                d = self._get_or_load_chat_turns(active_chat_id)
                return d[-1] if d else None
            if self._is_project_mode(mode, project_id):
                assert project_id is not None
                turns = self._get_or_load_project_turns(project_id)
                return turns[-1] if turns else None
            return self._casual_turns[-1] if self._casual_turns else (self._turns[-1] if self._turns else None)

    def get_recent_turns(self, n: int = 3, mode: str = "casual", project_id: Optional[str] = None, chat_id: Optional[str] = None) -> List[SessionTurnRecord]:
        with self._lock:
            active_chat_id, active_temp, active_pid = self._resolve_active_ctx(mode=mode, project_id=project_id, chat_id=chat_id)
            if active_temp:
                return list(self._temporary_turns)[-n:]
            if active_chat_id:
                d = self._get_or_load_chat_turns(active_chat_id)
                return list(d)[-n:]
            if self._is_project_mode(mode, project_id):
                assert project_id is not None
                turns = self._get_or_load_project_turns(project_id)
                return list(turns)[-n:]
            src = self._casual_turns if self._casual_turns else list(self._turns)
            return src[-n:]

    def get_recent_history_context(
        self,
        n: int = 3,
        max_chars_per_turn: int = 500,
        mode: str = "casual",
        project_id: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> str:
        """
        Returns a compact conversation history buffer of the last n turns
        to give Tier 2/3 stateless APIs short-term conversational continuity.
        """
        with self._lock:
            active_chat_id, active_temp, active_pid = self._resolve_active_ctx(mode=mode, project_id=project_id, chat_id=chat_id)
            # Return empty string for saved/active chat_id or project mode sessions to prevent duplicate history injection
            if active_chat_id or self._is_project_mode(mode, project_id) or active_pid:
                return ""

            if active_temp:
                turns_source = list(self._temporary_turns)
            else:
                turns_source = self._casual_turns if self._casual_turns else list(self._turns)

            turns = [t for t in turns_source if not ATTRIBUTION_PATTERN.search(t.prompt)][-n:]
            if not turns:
                return ""
            blocks = []
            for t in turns:
                resp = t.response or ""
                if "<!doctype html" in resp.lower():
                    resp = re.sub(
                        r"```html[\s\S]*?```",
                        "[Complete HTML Application Artifact Rendered]",
                        resp,
                        flags=re.IGNORECASE,
                    )
                clean_resp = resp.strip()
                if len(clean_resp) > max_chars_per_turn:
                    clean_resp = clean_resp[:max_chars_per_turn].rsplit(" ", 1)[0] + "..."
                blocks.append(f"User: {t.prompt}\nAssistant: {clean_resp}")
            return "\n\n".join(blocks)

    def get_chat_history_messages(
        self,
        max_turns: int = HISTORY_EXCHANGES,
        max_chars: int = HISTORY_MAX_CHARS,
        mode: str = "casual",
        project_id: Optional[str] = None,
        provider: Optional[str] = None,
        chat_id: Optional[str] = None,
        temporary: Optional[bool] = None,
    ) -> List[Dict[str, str]]:
        """
        Returns a list of {"role": "user", "content": ...} and {"role": "assistant", "content": ...}
        dictionaries for up to max_turns turns.
        Replaces massive raw HTML blocks in history with "[Generated Single-File HTML Application]"
        so token budgets stay lean while preserving full conversational text.
        """
        messages: List[Dict[str, str]] = []
        with self._lock:
            effective_max_turns = max_turns
            if provider and max_turns == HISTORY_EXCHANGES:
                clean_p = provider.lower().strip()
                if clean_p in PROVIDER_EXCHANGE_OVERRIDE:
                    effective_max_turns = PROVIDER_EXCHANGE_OVERRIDE[clean_p]
                elif ":" in clean_p:
                    for part in clean_p.split(":"):
                        part = part.strip()
                        if part in PROVIDER_EXCHANGE_OVERRIDE:
                            effective_max_turns = PROVIDER_EXCHANGE_OVERRIDE[part]
                            break

            active_chat_id, active_temp, active_pid = self._resolve_active_ctx(
                mode=mode, project_id=project_id, chat_id=chat_id, temporary=temporary
            )

            if active_temp:
                candidate_turns = [t for t in self._temporary_turns if not ATTRIBUTION_PATTERN.search(t.prompt)][-effective_max_turns:]
                collected_pairs: List[List[Dict[str, str]]] = []
                accumulated_chars = 0
                for t in reversed(candidate_turns):
                    u_text = t.prompt.strip()
                    resp = (t.response or "").strip()
                    if "<!doctype html" in resp.lower():
                        resp = re.sub(r"```html[\s\S]*?```", "[Generated Single-File HTML Application]", resp, flags=re.IGNORECASE)
                        resp = re.sub(r"<!DOCTYPE html[\s\S]*?</html>", "[Generated Single-File HTML Application]", resp, flags=re.IGNORECASE)
                    if len(resp) > REPLY_MAX_CHARS:
                        resp = resp[:REPLY_MAX_CHARS].rsplit(" ", 1)[0] + "..."

                    turn_chars = len(u_text) + len(resp)
                    if collected_pairs and (accumulated_chars + turn_chars > max_chars):
                        break

                    pair = [{"role": "user", "content": u_text}]
                    if resp:
                        pair.append({"role": "assistant", "content": resp})
                    collected_pairs.append(pair)
                    accumulated_chars += turn_chars

                collected_pairs.reverse()
                for pair in collected_pairs:
                    messages.extend(pair)
                return messages

            if active_chat_id:
                from chat_store import chat_store
                safe_cid = active_chat_id.strip().lower()

                # Prepend context pair if inside a project
                if active_pid:
                    safe_pid = sanitize_project_id(active_pid)
                    instr = chat_store.read_instructions(safe_pid)
                    pins_list = chat_store.list_pins(safe_pid)
                    pins_text = "\n".join(f"- {p}" for p in pins_list)[:600] if pins_list else "(None)"
                    recap_data = chat_store.read_recap(safe_cid)
                    recap_lines = recap_data.get("lines", [])
                    recap_text = "\n".join(f"- {r}" for r in recap_lines)[:RECAP_MAX] if recap_lines else "(None)"

                    proj_context = (
                        f"[PROJECT CONTEXT]\n"
                        f"INSTRUCTIONS:\n{instr or '(None)'}\n\n"
                        f"PINNED DECISIONS:\n{pins_text}\n\n"
                        f"PROJECT RECAP:\n{recap_text}"
                    )
                    messages.append({"role": "user", "content": proj_context})
                    messages.append({"role": "assistant", "content": "Understood."})
                else:
                    recap_data = chat_store.read_recap(safe_cid)
                    recap_lines = recap_data.get("lines", [])
                    if recap_lines:
                        recap_text = "\n".join(f"- {r}" for r in recap_lines)[:RECAP_MAX]
                        messages.append({"role": "user", "content": f"[RECAP]\n{recap_text}"})
                        messages.append({"role": "assistant", "content": "Understood."})

                chat_turns = self._get_or_load_chat_turns(safe_cid)
                valid_turns: List[SessionTurnRecord] = []
                for t in chat_turns:
                    if t.response == "[Context Reset]" or t.turn_id == 0:
                        valid_turns = []
                    else:
                        valid_turns.append(t)

                candidate_turns = [t for t in valid_turns if not ATTRIBUTION_PATTERN.search(t.prompt)][-effective_max_turns:]
                collected_pairs = []
                accumulated_chars = 0
                for t in reversed(candidate_turns):
                    u_text = t.prompt.strip()
                    resp = (t.response or "").strip()
                    if "<!doctype html" in resp.lower():
                        resp = re.sub(r"```html[\s\S]*?```", "[Generated Single-File HTML Application]", resp, flags=re.IGNORECASE)
                        resp = re.sub(r"<!DOCTYPE html[\s\S]*?</html>", "[Generated Single-File HTML Application]", resp, flags=re.IGNORECASE)
                    if len(resp) > REPLY_MAX_CHARS:
                        resp = resp[:REPLY_MAX_CHARS].rsplit(" ", 1)[0] + "..."

                    turn_chars = len(u_text) + len(resp)
                    if collected_pairs and (accumulated_chars + turn_chars > max_chars):
                        break

                    pair = [{"role": "user", "content": u_text}]
                    if resp:
                        pair.append({"role": "assistant", "content": resp})
                    collected_pairs.append(pair)
                    accumulated_chars += turn_chars

                collected_pairs.reverse()
                for pair in collected_pairs:
                    messages.extend(pair)
                return messages

            if self._is_project_mode(mode, project_id):
                assert project_id is not None
                safe_pid = self._sanitize_project_id(project_id)
                proj_dir = self.get_project_dir(safe_pid)

                # 1. Read instructions (max 2000 chars)
                instr_file = proj_dir / "instructions.txt"
                instr_text = instr_file.read_text(encoding="utf-8").strip()[:2000] if instr_file.exists() else ""

                # 2. Read pins (max 600 chars)
                pins_list = self.list_pins(safe_pid)
                pins_text = "\n".join(f"- {p}" for p in pins_list)[:600] if pins_list else "(None)"

                # 3. Read recap (max 800 chars)
                recap_list = self.get_recap_entries(safe_pid)
                recap_text = "\n".join(f"- {r}" for r in recap_list)[:800] if recap_list else "(None)"

                context_blocks = [
                    f"[PROJECT CONTEXT & INSTRUCTIONS]\n{instr_text or '(None)'}",
                    f"[PINNED DECISIONS]\n{pins_text}",
                    f"[PROJECT RECAP]\n{recap_text}",
                ]
                context_msg = "\n\n".join(context_blocks)
                messages.append({"role": "system", "content": context_msg})

                turns_source = self._get_or_load_project_turns(safe_pid)
            else:
                if self._casual_recap:
                    recap_text = "\n".join(self._casual_recap)[:800]
                    messages.append({"role": "system", "content": f"[CASUAL SESSION RECAP]\n{recap_text}"})

                turns_source = self._casual_turns if self._casual_turns else self._turns

            candidate_turns = [t for t in turns_source if not ATTRIBUTION_PATTERN.search(t.prompt)][-effective_max_turns:]
            collected_pairs = []
            accumulated_chars = 0

            for t in reversed(candidate_turns):
                u_text = t.prompt.strip()
                resp = (t.response or "").strip()
                if "<!doctype html" in resp.lower():
                    resp = re.sub(
                        r"```html[\s\S]*?```",
                        "[Generated Single-File HTML Application]",
                        resp,
                        flags=re.IGNORECASE,
                    )
                    resp = re.sub(
                        r"<!DOCTYPE html[\s\S]*?</html>",
                        "[Generated Single-File HTML Application]",
                        resp,
                        flags=re.IGNORECASE,
                    )
                if len(resp) > REPLY_MAX_CHARS:
                    resp = resp[:REPLY_MAX_CHARS].rsplit(" ", 1)[0] + "..."

                turn_chars = len(u_text) + len(resp)
                if collected_pairs and (accumulated_chars + turn_chars > max_chars):
                    break

                pair = [{"role": "user", "content": u_text}]
                if resp:
                    pair.append({"role": "assistant", "content": resp})

                collected_pairs.append(pair)
                accumulated_chars += turn_chars

            collected_pairs.reverse()
            for pair in collected_pairs:
                messages.extend(pair)

        return messages

    def get_last_artifact_turn(
        self,
        mode: str = "casual",
        project_id: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> Optional[SessionTurnRecord]:
        with self._lock:
            active_chat_id, active_temp, active_pid = self._resolve_active_ctx(mode=mode, project_id=project_id, chat_id=chat_id)
            if active_temp:
                for rec in reversed(self._temporary_turns):
                    if rec.has_html_artifact and rec.html_code:
                        return rec
                return None

            if active_chat_id:
                from chat_store import chat_store
                safe_cid = active_chat_id.strip().lower()
                if safe_cid in self._chat_lru:
                    for rec in reversed(self._chat_lru[safe_cid]):
                        if rec.has_html_artifact and rec.html_code:
                            return rec
                art = chat_store.load_artifact(safe_cid)
                if art:
                    html, note = art
                    return SessionTurnRecord(
                        turn_id=0,
                        prompt="[Chat Saved Artifact]",
                        target_tier="chat",
                        actual_provider="chat",
                        has_html_artifact=True,
                        html_code=html,
                        response=html,
                    )
                return None

            if self._is_project_mode(mode, project_id):
                assert project_id is not None
                safe_pid = self._sanitize_project_id(project_id)
                proj_dir = self.get_project_dir(safe_pid)
                artifact_file = proj_dir / "last_app.html"
                if artifact_file.exists():
                    code = artifact_file.read_text(encoding="utf-8").strip()
                    if code:
                        turns = self._get_or_load_project_turns(safe_pid)
                        for rec in reversed(turns):
                            if rec.has_html_artifact and rec.html_code:
                                return rec
                        return SessionTurnRecord(
                            turn_id=0,
                            prompt="[Project Saved Artifact]",
                            target_tier="project",
                            actual_provider="project",
                            has_html_artifact=True,
                            html_code=code,
                            response=code,
                        )
                turns = self._get_or_load_project_turns(safe_pid)
                for rec in reversed(turns):
                    if rec.has_html_artifact and rec.html_code:
                        return rec
                return None
            else:
                src = self._casual_turns if self._casual_turns else self._turns
                for rec in reversed(src):
                    if rec.has_html_artifact and rec.html_code:
                        return rec
                return None

    def format_attribution_answer(
        self,
        query: str = "",
        mode: str = "casual",
        project_id: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> str:
        last: Optional[SessionTurnRecord] = None
        with self._lock:
            active_chat_id, active_temp, active_pid = self._resolve_active_ctx(mode=mode, project_id=project_id, chat_id=chat_id)
            if active_temp:
                turns_source = list(self._temporary_turns)
            elif active_chat_id:
                turns_source = list(self._get_or_load_chat_turns(active_chat_id))
            elif self._is_project_mode(mode, project_id):
                assert project_id is not None
                turns_source = list(self._get_or_load_project_turns(project_id))
            else:
                turns_source = self._casual_turns if self._casual_turns else self._turns

            if query:
                q_lower = query.lower()
                stop_words = {"for", "the", "which", "what", "model", "did", "you", "use", "was", "used", "made", "built", "generate", "generated"}
                words = [w for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", q_lower) if w not in stop_words]
                if words:
                    for rec in reversed(turns_source):
                        if any(w in rec.prompt.lower() for w in words):
                            last = rec
                            break

            if not last:
                last = turns_source[-1] if turns_source else None

        if not last:
            return (
                "No previous deliverable has been generated in this session yet. "
                "AIVCO routes frontend architecture to Claude/ChatGPT Web Bridge and fast logic/API tasks to Groq (Llama 3.3 70B) and Gemini Pro."
            )
        friendly = self.PROVIDER_DISPLAY_NAMES.get(
            last.actual_provider, last.actual_provider.upper()
        )
        if last.cascaded_from and last.cascaded_from != last.actual_provider:
            orig = self.PROVIDER_DISPLAY_NAMES.get(
                last.cascaded_from, last.cascaded_from.upper()
            )
            return (
                f"For your previous request (`\"{last.prompt[:80]}\"`), AIVCO initially targeted **{orig}**, "
                f"detected a rate limit/cooldown, and automatically cascaded execution to **{friendly}**."
            )
        return (
            f"For your previous request (`\"{last.prompt[:80]}\"`), AIVCO used **{friendly}**."
        )

    def clear(self, mode: str = "casual", project_id: Optional[str] = None, chat_id: Optional[str] = None) -> None:
        with self._lock:
            active_chat_id, active_temp, active_pid = self._resolve_active_ctx(mode=mode, project_id=project_id, chat_id=chat_id)
            if active_temp:
                self._temporary_turns.clear()
                return
            if active_chat_id:
                from chat_store import chat_store
                safe_cid = active_chat_id.strip().lower()
                chat_store.append_turn(safe_cid, {"type": "context_reset", "ts": time.time()})
                if safe_cid in self._chat_lru:
                    self._chat_lru[safe_cid].clear()
                return

            if self._is_project_mode(mode, project_id):
                assert project_id is not None
                safe_pid = self._sanitize_project_id(project_id)
                if safe_pid in self._project_ledgers:
                    self._project_ledgers[safe_pid].clear()
            else:
                self._turns.clear()
                self._casual_turns.clear()
                self._casual_recap.clear()
                self._turn_counter = 0


session_ledger = SessionExecutionLedger()


# ── Live Provider Health & Circuit Breaker ───────────────────────────────────

class ProviderHealthRecord:
    def __init__(self, name: str):
        self.name = name
        self.healthy: bool = True
        self.cooldown_until: float = 0.0
        self.consecutive_failures: int = 0
        self.avg_latency_ms: float = 0.0
        self.last_failure_reason: str = ""

    def is_available(self, extension_connected: bool = True, now: Optional[float] = None) -> bool:
        if self.name in WEB_PROVIDERS and not extension_connected:
            return False
        current_time = now if now is not None else time.time()
        if current_time < self.cooldown_until:
            return False
        return True


class ProviderHealthRegistry:
    """
    Thread-safe Live Provider Health & Circuit Breaker with Exponential Backoff
    and Rolling Latency tie-breaking.
    """

    DEFAULT_PROVIDERS = [
        "claude",
        "chatgpt",
        "gemini_web",
        "grok",
        "gemini",
        "groq",
        "local_llama",
    ]

    def __init__(self):
        self._lock = threading.Lock()
        self._registry: Dict[str, ProviderHealthRecord] = {
            p: ProviderHealthRecord(p) for p in self.DEFAULT_PROVIDERS
        }
        self._has_active_cooldown: bool = False

    def _normalize(self, provider: str) -> str:
        prov = (provider or "").strip().lower().replace("-", "_")
        if prov in WEB_PROVIDERS:
            return prov
        if ":" in prov:
            prov = prov.split(":", 1)[1].strip()
        return KNOWN_PROVIDER_ALIASES.get(prov, prov)

    def get_record(self, provider: str) -> ProviderHealthRecord:
        norm = self._normalize(provider)
        with self._lock:
            if norm not in self._registry:
                self._registry[norm] = ProviderHealthRecord(norm)
            return self._registry[norm]

    def record_failure(self, provider: str, reason: str = "") -> None:
        norm = self._normalize(provider)
        reason_lower = (reason or "").lower()
        now = time.time()
        with self._lock:
            if norm not in self._registry:
                self._registry[norm] = ProviderHealthRecord(norm)
            rec = self._registry[norm]
            rec.consecutive_failures += 1
            rec.last_failure_reason = reason
            rec.healthy = False
            self._has_active_cooldown = True

            # Exponential backoff scaling capped at 30 minutes (1800s)
            multiplier = min(3, 2 ** max(0, rec.consecutive_failures - 1))
            if any(k in reason_lower for k in ("rate_limit", "rate limit", "cloudflare", "quota", "429")):
                base_cd = 600.0  # 10 min base for rate limit
                rec.cooldown_until = now + min(1800.0, base_cd * multiplier)
            elif any(k in reason_lower for k in ("timeout", "disconnected", "disconnect", "unavailable")):
                base_cd = 120.0  # 2 min base for timeout/bridge drop
                rec.cooldown_until = now + min(600.0, base_cd * multiplier)
            else:
                base_cd = 60.0 if rec.consecutive_failures >= 2 else 15.0
                rec.cooldown_until = now + min(300.0, base_cd * multiplier)

    def record_success(self, provider: str, latency_ms: float = 0.0) -> None:
        norm = self._normalize(provider)
        now = time.time()
        with self._lock:
            if norm not in self._registry:
                self._registry[norm] = ProviderHealthRecord(norm)
            rec = self._registry[norm]
            rec.consecutive_failures = 0
            rec.cooldown_until = 0.0
            rec.healthy = True
            rec.last_failure_reason = ""
            if latency_ms > 0:
                rec.avg_latency_ms = (
                    latency_ms
                    if rec.avg_latency_ms <= 0
                    else (0.7 * rec.avg_latency_ms + 0.3 * latency_ms)
                )
            self._has_active_cooldown = any(
                now < r.cooldown_until for r in self._registry.values()
            )

    def is_available(self, provider: str, extension_connected: bool = True) -> bool:
        norm = self._normalize(provider)
        if not self._has_active_cooldown and (norm not in WEB_PROVIDERS or extension_connected):
            return True
        now = time.time()
        with self._lock:
            rec = self._registry.setdefault(norm, ProviderHealthRecord(norm))
            return rec.is_available(extension_connected=extension_connected, now=now)

    def get_best_available_provider(
        self,
        preferred_chain: List[str],
        extension_connected: bool = True,
    ) -> str:
        if not preferred_chain:
            return "groq"

        first_norm = self._normalize(preferred_chain[0])
        if not self._has_active_cooldown and (first_norm not in WEB_PROVIDERS or extension_connected):
            return first_norm

        now = time.time()
        with self._lock:
            has_any_cd = False
            available_candidates: List[str] = []

            for p in preferred_chain:
                norm = self._normalize(p)
                rec = self._registry.setdefault(norm, ProviderHealthRecord(norm))

                if norm in WEB_PROVIDERS and not extension_connected:
                    continue

                if now < rec.cooldown_until:
                    has_any_cd = True
                    continue

                available_candidates.append(norm)

            self._has_active_cooldown = any(
                now < r.cooldown_until for r in self._registry.values()
            )

            if available_candidates:
                return available_candidates[0]

            # All preferred candidates are in cooldown or disconnected:
            # Pick the non-disconnected candidate whose cooldown expires soonest.
            fallback_pool = [
                self._normalize(p)
                for p in preferred_chain
                if not (self._normalize(p) in WEB_PROVIDERS and not extension_connected)
            ]
            if fallback_pool:
                return min(
                    fallback_pool,
                    key=lambda x: self._registry[x].cooldown_until,
                )

        return "groq"

    def reset_all(self) -> None:
        with self._lock:
            for rec in self._registry.values():
                rec.healthy = True
                rec.cooldown_until = 0.0
                rec.consecutive_failures = 0
                rec.last_failure_reason = ""
            self._has_active_cooldown = False


provider_health = ProviderHealthRegistry()


# ── Fuzzy Provider Directive Extraction ──────────────────────────────────────

DIRECTIVE_TRIGGER_CHECK = re.compile(
    r"\b(?:use|ask|tell|run|search|via|with|on|in|switch\s+to|@)\b|^[a-zA-Z0-9_\-]+:\s*",
    re.IGNORECASE,
)

TRIGGER_PATTERN = re.compile(
    r"^(?:(?:use|ask|tell|run|search|via|with|on|in)\s+)?@?([a-zA-Z0-9_\-]+(?:\s+(?:web|browser|api|key|pro|flash))?)(?:\s*(?:to\s+(?:search\s+)?|for\s+|:\s*|\s+-\s*|\s+–\s*)|\s+)(.*)$",
    re.IGNORECASE,
)

SAFE_MERGED_TO_PREFIXES = frozenset(
    {"claude", "cladue", "claud", "cloude", "calude", "chatgpt", "chagpt", "chatgtp", "gpt", "gemini", "gemeni", "gmini", "grok", "gork", "groq"}
)


def match_provider_token(token: str) -> Optional[str]:
    """Matches a token to a canonical provider name using exact aliases and safe SequenceMatcher."""
    t = token.lower().strip()
    if not t:
        return None
    if t in KNOWN_PROVIDER_ALIASES:
        return KNOWN_PROVIDER_ALIASES[t]

    # Guard against common English words ending in 'to' (like 'photo', 'auto', 'into', 'onto')
    if t.endswith("to") and len(t) >= 5:
        stem = t[:-2]
        if stem in SAFE_MERGED_TO_PREFIXES:
            return KNOWN_PROVIDER_ALIASES.get(stem)

    lt = len(t)
    if lt < 3 or lt > 12 or t[0] not in ("c", "g", "o", "x", "l", "k"):
        return None

    best_can: Optional[str] = None
    best_ratio: float = 0.0
    for can, targets in CANONICAL_PROVIDERS.items():
        for target in targets:
            if abs(lt - len(target)) <= 2:
                ratio = difflib.SequenceMatcher(None, t, target).ratio()
                if ratio >= 0.78 and ratio > best_ratio:
                    best_ratio = ratio
                    best_can = can
    return best_can


def extract_fuzzy_provider_directive(prompt: str) -> Tuple[Optional[str], str]:
    """
    Sub-millisecond fuzzy typo-tolerant provider directive extraction.
    Detects provider commands at the start or end of a prompt and strips ONLY the routing prefix/suffix.
    """
    raw = (prompt or "").strip()
    if not raw or not DIRECTIVE_TRIGGER_CHECK.search(raw):
        return None, raw

    # Check 1: Explicit switch commands ("switch to claude", "switch to grok")
    m_switch = re.match(r"^(?:switch\s+to\s+)([\w\s\-]+?)[.?!]*$", raw, re.IGNORECASE)
    if m_switch:
        prov = match_provider_token(m_switch.group(1))
        if prov:
            return prov, ""

    # Check 2: Prefix directives: (use|ask|tell|run|via|with|search)? @?(provider)(to|:|-) rest
    m_pref = TRIGGER_PATTERN.match(raw)
    if m_pref:
        token = m_pref.group(1).strip()
        rest = m_pref.group(2).strip()
        prov = match_provider_token(token)
        if prov:
            return prov, rest

    # Check 3: Short standalone ("use cladue", "ask gpt", "chagpt")
    m_standalone = re.match(
        r"^(?:(?:use|ask|tell)\s+)?@?([a-zA-Z0-9_\-]+(?:\s+(?:web|browser|api|key|pro|flash))?)[.?!]*$",
        raw,
        re.IGNORECASE,
    )
    if m_standalone:
        token = m_standalone.group(1).strip()
        prov = match_provider_token(token)
        if prov:
            return prov, ""

    # Check 4: Inline trailing directive ("... using cladue", "... with chagpt")
    m_inline = re.search(
        r"\b(?:using|with|via|in|on)\s+@?([a-zA-Z0-9_\-]+(?:\s+(?:web|browser|api|key|pro|flash))?)[.?!]*$",
        raw,
        re.IGNORECASE,
    )
    if m_inline:
        candidate_token = m_inline.group(1).strip()
        prov = match_provider_token(candidate_token)
        if prov:
            clean = raw[: m_inline.start()].strip()
            return prov, clean

    return None, raw


# ── Domain Feature Blueprints ────────────────────────────────────────────────

VIDEO_STREAMING_BLUEPRINT = """DOMAIN FEATURE BLUEPRINT (VIDEO & STREAMING PLATFORM):
- Hero Billboard: Large cinematic backdrop with title logo, metadata badges (4K Ultra HD, Dolby Vision, Match 98%), summary synopsis, and interactive "Play Trailer" modal embed.
- Horizontal Content Rails: Multiple smooth-scrolling category carousels (Trending Now, Top 10 Today, Originals, Action & Sci-Fi) with hover scale-up preview cards and badge overlays.
- Real-Time Search & Filtering: Instant search input with live genre filter chips that dynamically filter the video catalog with zero page reload.
- Watchlist & Favorites Persistence: Interactive "Add to Watchlist" / "Like" buttons that persist user selections in browser localStorage.
- Responsive Video Player Modal: Fullscreen-capable modal with working playback scrubber, time counter, volume slider, play/pause toggles, and ESC key listener.
- Aesthetic: Dark obsidian cinema palette (#141414 base, #E50914 red accents or crimson gradients), custom inline SVG logos/icons, zero external image broken links."""

MUSIC_AUDIO_BLUEPRINT = """DOMAIN FEATURE BLUEPRINT (MUSIC & AUDIO STREAMING SYSTEM):
- 3-Pane Architecture: Collapsible left navigation sidebar (Home, Search, Your Library, Create Playlist), central scrollable track table with album artwork & duration, and sticky bottom playback control bar.
- Real Web Audio API Sound Engine: Built-in JavaScript Web Audio API synthesizer / oscillator sound generator so tracks actually play real synthesized melodic chords/beats with visual feedback without needing external MP3 URLs!
- Interactive Playback Controls: Play/pause toggle, next/previous track, seek scrubber with live time counter, volume slider, shuffle and repeat toggles.
- Live Equalizer & Visualizer: Animated HTML5 canvas or CSS bar equalizer reacting in real-time to track playback.
- Playlist & Queue Management: Functional playlist switching, queue drawer, and track favoriting stored in localStorage.
- Aesthetic: Spotify-grade dark jade & obsidian theme (#121212, #1DB954 vibrant green accents), modern typography, custom inline SVG icons."""

ECOMMERCE_BLUEPRINT = """DOMAIN FEATURE BLUEPRINT (EDITORIAL E-COMMERCE STOREFRONT):
- Editorial Product Grid: Responsive multi-column showcase with category filtering, price/popularity sorting, and badges (Bestseller, Limited Edition).
- Bespoke Visuals: Elegant custom inline SVG product artwork/graphics for every item with hover image switching.
- Slide-Over Cart Drawer: Live slide-in cart with quantity increment/decrement, line-item removal, real-time subtotal/tax calculation, and promo code entry.
- Free Shipping Progress Bar: Dynamic animated meter calculating "$X away from Free Express Shipping".
- Interactive Checkout & Feedback: Mock checkout modal with validation and instant floating toast notifications upon "Added to Bag".
- Aesthetic: Luxury editorial minimalism or sleek dark glassmorphism, refined serif/sans typography, smooth CSS transitions."""

COMMAND_CENTER_BLUEPRINT = """DOMAIN FEATURE BLUEPRINT (AIVCO COMMAND CENTER & DEVELOPER PORTFOLIO):
- Executive Hero Header: Dynamic system status badge (Kernel v2.0 Operational), animated greeting, and operator profile summary (Parikshit Koirala / AIvco).
- Live System Telemetry Deck: Interactive real-time metrics (CPU latency simulation, RAM efficiency counters, model arbitration graph).
- Active Project Showcase: Interactive cards for active engineering initiatives (AIVCO Cognitive Kernel, Multi-Layer Memory Engine, Chrome Extension Bridge) with tech stack tags and live status badges.
- Interactive Task Kanban / Tracker: Draggable/clickable task board (To Do, In Progress, Shipped) with ability to add new tasks locally in localStorage.
- Aesthetic: High-tech obsidian glassmorphism (#0A0F1D, electric cyan #00F0FF and violet #8B5CF6 glow accents), backdrop-filter blur, custom inline SVG telemetry icons."""


def _is_true_frontend_intent(text: str) -> bool:
    """
    Ensures backend/desktop requests like 'write a python tkinter gui app' or 'build a fastapi app'
    are NOT misclassified as single-file HTML websites unless HTML/web is explicitly requested.
    """
    lower = text.lower()
    if BACKEND_SCRIPT_GUARD.search(lower):
        explicit_web = any(
            w in lower
            for w in (
                "html",
                "website",
                "webasite",
                "web app",
                "webpage",
                "css",
                "three.js",
                "ecommerce",
                "e-commerce",
                "ecommerice",
                "netflix",
                "nelflix",
                "spotify",
                "youtube",
                "landing page",
                "storefront",
                "ui only",
            )
        )
        return explicit_web
    return bool(FRONTEND_PATTERN.search(text) or KernelCognitiveArbiter._FAST_WEB.search(text))


def synthesize_elite_prompt(
    raw_intent: str,
    target_tier: str = "",
    memory_context: str = "",
    mode: str = "casual",
    project_id: Optional[str] = None,
) -> str:
    """
    Transforms casual or typo-laden operator requests into elite, production-grade
    engineering specifications before dispatching to AI providers.
    """
    clean_intent = (raw_intent or "").strip()
    if not clean_intent:
        return clean_intent

    tier_norm = (target_tier or "").strip().lower()
    if tier_norm == "kernel":
        return clean_intent

    # 1. Identity / Personal queries
    if IDENTITY_PATTERN.search(clean_intent) and not META_CONVERSATION_EXCLUDE.search(clean_intent):
        return (
            "Analyze the provided Operator Context in the Executive Brief and provide a comprehensive, "
            "structured executive summary of who the operator (Parikshit Koirala / AIvco) is, including "
            "technical stack, active engineering projects, academic background, and design preferences."
        )

    # 2. Iterative Refinement on active HTML artifact
    last_artifact = session_ledger.get_last_artifact_turn(mode=mode, project_id=project_id)
    if (
        last_artifact
        and ITERATIVE_REFINEMENT_PATTERN.search(clean_intent)
        and len(clean_intent) < 300
    ):
        return (
            f"ITERATIVE REFINEMENT DIRECTIVE: {clean_intent}\n\n"
            "PREVIOUS ACTIVE APPLICATION BASELINE:\n"
            f"```html\n{last_artifact.html_code[:12000]}\n```\n\n"
            "EXECUTIVE AUTONOMY MANDATE:\n"
            "- Preserve all working state, styles, and features from the baseline above.\n"
            "- Apply the requested modification cleanly and output the COMPLETE updated `<!DOCTYPE html>` file inside a single ```html code block."
        )

    # 3. Frontend / Web App / UI / Clone queries
    if _is_true_frontend_intent(clean_intent):
        lower_intent = clean_intent.lower()
        if re.search(
            r"\b(netflix|nelflix|youtube|video|stream(?:ing)?|cinema|movie)\b",
            lower_intent,
        ):
            blueprint = VIDEO_STREAMING_BLUEPRINT
        elif re.search(
            r"\b(spotify|music|audio|player|song|sound|playlist|track)\b",
            lower_intent,
        ):
            blueprint = MUSIC_AUDIO_BLUEPRINT
        elif re.search(
            r"\b(ecommerce|e-?commer(?:ce|ice)|store|shop|brand|sell|tshimhchet|cart|checkout|product)\b",
            lower_intent,
        ):
            blueprint = ECOMMERCE_BLUEPRINT
        else:
            blueprint = COMMAND_CENTER_BLUEPRINT

        return (
            f"USER OBJECTIVE: {clean_intent}\n\n"
            f"{blueprint}\n\n"
            "EXECUTIVE AUTONOMY MANDATE (ZERO HAND-HOLDING):\n"
            "1. NEVER ask the operator clarifying questions (e.g., never ask 'which framework?', 'what platform?', or 'would you like starter code?').\n"
            "2. NEVER reply with a high-level text roadmap, folder tree scaffold, or architecture outline when the operator asks to build, make, or clone an app, website, or UI.\n"
            "3. Act as an autonomous, decisive CTO/CEO: immediately choose the best single-file HTML5/CSS3/JS architecture and output the COMPLETE, fully functional, self-contained `<!DOCTYPE html>` application inside a single ```html code block so it renders immediately in AIVCO's Live Preview drawer.\n"
            "4. Write 100% pure vanilla JavaScript inside `<script>` tags—NEVER output TypeScript syntax (`interface`, `type X =`, `as HTMLElement`, or `: number` parameter types) inside an HTML `<script>` block.\n"
            "5. Even if the operator writes typos (e.g., 'claong website of nelflix', 'buikd', 'amek'), infer the exact target product and build the complete working clone on the first turn.\n\n"
            "ELITE ENGINEERING SPECIFICATION:\n"
            "- Architect and output a complete, self-contained, single-file `<!DOCTYPE html>` application (HTML5 + modern CSS3 + vanilla JavaScript).\n"
            "- Design Aesthetic: Bespoke, human-crafted editorial or dark obsidian/glassmorphic UI with custom inline SVG artwork, micro-interactions, responsive CSS Grid/Flexbox layout, and zero broken external image links.\n"
            "- Functional Completeness: Implement working state management, interactive filters/search/controls, smooth drawers/modals, and realistic domain data.\n"
            "- Output Contract: Do NOT use the Artifacts sandbox tool or external file cards. Output the ENTIRE runnable code directly inline inside a single ```html ... ``` fenced code block."
        )

    # 4. Coding / Script / Backend queries
    if CODING_PATTERN.search(clean_intent):
        return (
            f"USER OBJECTIVE: {clean_intent}\n\n"
            "EXECUTIVE AUTONOMY MANDATE (ZERO HAND-HOLDING):\n"
            "- Do NOT ask clarifying questions or output empty folder scaffolds.\n"
            "- Deliver complete, production-grade, well-structured, and immediately runnable code inside standard markdown fenced code blocks.\n"
            "- Handle edge cases, include clear inline documentation, and provide a concise architectural breakdown.\n\n"
            "ELITE ENGINEERING SPECIFICATION:\n"
            "- Deliver clean, production-ready code with complete logic and zero placeholders.\n"
            "- Output immediately runnable script in standard markdown fenced code blocks."
        )

    # 5. Real-Time / News / Sports / Temporal queries
    if LIVE_SEARCH_PATTERN.search(clean_intent):
        return (
            f"USER OBJECTIVE: {clean_intent}\n\n"
            "ANTI-HALLUCINATION & TEMPORAL VERIFICATION MANDATE:\n"
            "- If you have grounded web search context provided, answer accurately citing the sources.\n"
            "- If you do NOT have live web access or grounded search results for real-time, current, or post-2024 events, "
            "you MUST explicitly state: 'I do not have live web access or real-time verification for post-2024 events.'\n"
            "- NEVER fabricate, guess, or invent sports matches, scores, player transfers, or news headlines."
        )

    return clean_intent


def compress_memory_context(memory_context: str, max_chars: int = 500) -> str:
    """
    Compresses multi-layer memory context into a dense, high-signal summary
    (max ~150 tokens) containing key operator directives, rules, and project context.
    """
    if not memory_context or not memory_context.strip():
        return "Operator: Parikshit Koirala (Sir/AIvco) | System: AIVCO Personal AI Architecture."

    lines = []
    for raw_line in memory_context.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        clean_line = re.sub(r"^[-*•✓\s]+", "", line).strip()
        if any(
            key in clean_line.lower()
            for key in (
                "operator address",
                "operating rules",
                "brevity protocol",
                "active project",
                "tech stack",
                "role",
                "preferred",
                "parikshit",
            )
        ):
            lines.append(clean_line)

    if not lines:
        for raw_line in memory_context.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                lines.append(re.sub(r"^[-*•✓\s]+", "", line).strip())
            if len(lines) >= 4:
                break

    summary = " | ".join(lines[:5])
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return summary or "Operator: Parikshit Koirala (Sir/AIvco) | System: AIVCO Personal AI Architecture."


def build_executive_envelope(
    prompt: str,
    target_tier: str,
    memory_context: str = "",
    mode: str = "casual",
    project_id: Optional[str] = None,
) -> str:
    """
    Wraps prompt with an Executive Context Envelope for Brain 4 Web AI models,
    and upgrades frontend/coding specifications for Tier 2/3 Cloud APIs.
    """
    tier_norm = (target_tier or "").lower()
    compact_memory_summary = compress_memory_context(memory_context)

    is_web_ai = (
        any(
            k in tier_norm
            for k in ("claude", "chatgpt", "gemini_web", "gemini-web", "grok", "brain4")
        )
        or tier_norm == "web"
    )
    if is_web_ai:
        synthesized_prompt = synthesize_elite_prompt(
            prompt, target_tier, memory_context, mode=mode, project_id=project_id
        )
        return (
            "[AIVCO EXECUTIVE BRIEF]\n"
            f"Operator Context: {compact_memory_summary}\n"
            "Execution Contract: Deliver production-grade, complete implementation. "
            "Act with absolute CEO autonomy (ZERO hand-holding, no clarifying questions, no text roadmaps). "
            "Do NOT use the Artifacts sandbox tool or create external file cards—output all code directly "
            "inline inside standard markdown fenced code blocks (e.g., ```html <!DOCTYPE html>... ```) "
            "so AIVCO's Live Preview Engine can render it immediately.\n"
            "[/AIVCO EXECUTIVE BRIEF]\n\n"
            "USER DIRECTIVE:\n"
            f"{synthesized_prompt}"
        )

    # For Tier 2/3 Cloud APIs (Groq/Gemini), keep simple prompts clean as required by unit tests,
    # unless the user is explicitly asking to build a Frontend Clone / Website or refining an active HTML artifact.
    is_refinement = bool(
        session_ledger.get_last_artifact_turn(mode=mode, project_id=project_id)
        and ITERATIVE_REFINEMENT_PATTERN.search(prompt)
    )
    if tier_norm in ("groq", "gemini") and (_is_true_frontend_intent(prompt) or is_refinement):
        return synthesize_elite_prompt(
            prompt, target_tier, memory_context, mode=mode, project_id=project_id
        )

    return prompt


# ── Multi-Vector Scoring & Capability Matrix ──────────────────────────────────

def compute_vector_scores(prompt: str) -> Dict[str, float]:
    """
    Computes 4-vector capability scores for incoming prompt in <0.1ms:
    - ui_density (0.0 to 1.0)
    - logic_depth (0.0 to 1.0)
    - context_need (0.0 to 1.0)
    - speed_priority (0.0 to 1.0)
    """
    clean = (prompt or "").strip().lower()
    if not clean:
        return {
            "ui_density": 0.0,
            "logic_depth": 0.0,
            "context_need": 0.0,
            "speed_priority": 1.0,
        }

    # 1. Context Need (Zero-LLM kernel queries & attribution queries)
    context_need = 0.0
    if (
        KernelCognitiveArbiter._FAST_KERNEL.search(clean)
        or IDENTITY_PATTERN.search(clean)
        or ATTRIBUTION_PATTERN.search(clean)
    ):
        context_need = 1.0
    elif any(
        k in clean
        for k in (
            "who am i",
            "my profile",
            "system status",
            "kernel status",
            "dataset stats",
            "tasks left",
            "clear screen",
            "clear memory",
        )
    ):
        context_need = 1.0

    # 2. UI Density (Full-stack HTML/JS apps, complete game engines, websites, dashboards, clones, or iterative refinements)
    ui_density = 0.0
    is_refinement = bool(
        session_ledger.get_last_artifact_turn()
        and ITERATIVE_REFINEMENT_PATTERN.search(clean)
    )
    if _is_true_frontend_intent(clean) or is_refinement:
        if is_refinement:
            ui_density = 0.90
        elif KernelCognitiveArbiter._FAST_WEB.search(clean):
            ui_density = 0.95
        elif any(
            k in clean
            for k in (
                "clone",
                "claong",
                "single-file",
                "complete website",
                "dashboard",
                "ecommerce",
                "store",
                "player",
                "game",
                "canvas",
                "three.js",
                "netflix",
                "nelflix",
                "youtube",
                "spotify",
            )
        ):
            ui_density = 0.90
        elif any(
            k in clean
            for k in (
                "website",
                "webasite",
                "web page",
                "html",
                "web app",
                "ui only",
                "working app",
            )
        ):
            ui_density = 0.80
        else:
            ui_density = 0.70

    # 3. Logic Depth (Algorithmic / Data Science / Python / SQL / Debugging)
    logic_depth = 0.0
    if KernelCognitiveArbiter._FAST_API.search(clean):
        logic_depth = 0.90
    elif CODING_PATTERN.search(clean):
        logic_depth = 0.85 if ui_density < 0.70 else 0.60
    elif any(
        k in clean
        for k in (
            "binary search",
            "algorithm",
            "regex",
            "sql",
            "fastapi",
            "refactor",
            "debug",
            "benchmark",
            "sort",
        )
    ):
        logic_depth = 0.80

    # 4. Speed Priority
    speed_priority = 0.2
    if len(clean) < 40 and ui_density < 0.70 and logic_depth < 0.70:
        speed_priority = 0.85
    elif "quick" in clean or "fast" in clean:
        speed_priority = 0.90

    return {
        "ui_density": round(ui_density, 2),
        "logic_depth": round(logic_depth, 2),
        "context_need": round(context_need, 2),
        "speed_priority": round(speed_priority, 2),
    }


# ── Kernel Cognitive Arbiter Class ───────────────────────────────────────────

class KernelCognitiveArbiter:
    # 1. Direct Kernel / System / Memory / Attribution Patterns (< 5ms resolution)
    KERNEL_PATTERNS = [
        r"^(who am i|my profile|system status|kernel status|dataset stats|tasks left)\??$",
        r"^(clear|reset|clean) (screen|memory|chat|context)$",
        r"^(open|launch) (terminal|code|browser|dashboard)$",
        r"^(what is|calc) \d+[\+\-\*\/]\d+",
        r"^(?:which|what)\s+(?:ai\s+)?model\s+(?:did\s+you\s+use|made\s+(?:this|that|it)|built\s+(?:this|that|it)|generated\s+(?:this|that|it))",
        r"\b(?:for\s+(?:the\s+)?[\w\s\-]+?\s*(?:clone|website|app|code)\s+which\s+model\s+did\s+you\s+use)\b",
    ]

    # 2. Brain 4 Web AI Patterns (Complex full-stack HTML/JS apps, complete game engines, massive code)
    WEB_AI_PATTERNS = [
        r"(single-file|complete|working)\s+(html|website|webasite|dashboard|game|canvas|three\.js|webpage|app)",
        r"(build|make|maek|create|write)\s+(?:me\s+)?(?:a|an)?\s*(?:(?:full|complete|interactive|tactical|working)\s+)*(?:clon(?:e|ing)|claong)?\s*(?:website|webasite|web\s*page|dashboard|clone|claong|app|store|e-?commer(?:ce|ice))",
        r"(build|make|maek|create|write)\s+(?:it|a\s+working\s+app|a\s+working\s+clone)",
        r"<!doctype html>",
        r"tactical dashboard|starting xi|probability calculator|pitch board",
        r"(spotify|netflix|nelflix|youtube|amazon)\s+(?:clon(?:e|ing)|claong)",
        r"(?:clon(?:e|ing)|claong)\s+(?:website|webasite|app)?\s*(?:of|for)?\s*(spotify|netflix|nelflix|youtube|amazon)",
        r"\b(?:ui\s+only|working\s+app)\b",
    ]

    # 3. Tier 2 Fast Cloud API Patterns (Groq Llama 70B / Gemini Pro)
    API_FAST_PATTERNS = [
        r"(write|implement|explain|debug|refactor) (python|c\+\+|javascript|rust|sql|function|script|regex)",
        r"(analyze|summarize|translate|extract|parse) (this|the following|json|data)",
        r"(compare|contrast|differences between)",
        r"\b(why|how does|architecture of)\b",
    ]

    _COMPILED_KERNEL = [re.compile(p, re.IGNORECASE) for p in KERNEL_PATTERNS]
    _COMPILED_WEB = [re.compile(p, re.IGNORECASE) for p in WEB_AI_PATTERNS]
    _COMPILED_API = [re.compile(p, re.IGNORECASE) for p in API_FAST_PATTERNS]

    _FAST_KERNEL = re.compile(
        "|".join(f"(?:{p})" for p in KERNEL_PATTERNS), re.IGNORECASE
    )
    _FAST_WEB = re.compile(
        "|".join(f"(?:{p})" for p in WEB_AI_PATTERNS), re.IGNORECASE
    )
    _FAST_API = re.compile(
        "|".join(f"(?:{p})" for p in API_FAST_PATTERNS), re.IGNORECASE
    )

    compute_vector_scores = staticmethod(compute_vector_scores)

    @classmethod
    def arbitrate(
        cls,
        prompt: str,
        target_override: Optional[str] = None,
        extension_connected: bool = True,
        log_telemetry: bool = True,
    ) -> Tuple[str, str]:
        """
        Deterministically arbitrates prompt intent in <5ms using multi-vector capability scoring
        and live provider health registry circuit breakers.
        """
        t0 = time.perf_counter()

        if target_override and target_override.strip().lower() not in [
            "auto",
            "auto (arbiter)",
            "",
        ]:
            tier = target_override
            rationale = f"[aivco-kernel] Manual override active: Routing to {target_override}."
        else:
            clean = (prompt or "").strip().lower()

            # Fast-path 1: Kernel-native or Model Attribution query (<0.005ms)
            if (cls._FAST_KERNEL.search(clean) or ATTRIBUTION_PATTERN.search(clean)) and not META_CONVERSATION_EXCLUDE.search(clean):
                tier = "kernel"
                rationale = ""
            else:
                prov_directive = None
                stripped_intent = prompt
                if DIRECTIVE_TRIGGER_CHECK.search(prompt):
                    prov_directive, stripped_intent = extract_fuzzy_provider_directive(
                        prompt
                    )

                if prov_directive:
                    if prov_directive == "kernel":
                        tier = "kernel"
                        rationale = ""
                    elif provider_health.is_available(
                        prov_directive, extension_connected=extension_connected
                    ):
                        tier = prov_directive
                        rationale = f"[aivco-kernel] Operator directive detected: Routing to {tier}..."
                    else:
                        scores = compute_vector_scores(stripped_intent or prompt)
                        chain = (
                            ["claude", "chatgpt", "gemini", "groq"]
                            if scores["ui_density"] >= 0.7
                            else ["groq", "gemini"]
                        )
                        tier = provider_health.get_best_available_provider(
                            chain, extension_connected=extension_connected
                        )
                        rationale = (
                            f"[aivco-kernel] Operator directive '{prov_directive}' on cooldown/unavailable. "
                            f"Auto-routed to {tier}."
                        )
                elif cls._FAST_WEB.search(clean) and _is_true_frontend_intent(clean):
                    chain = ["claude", "chatgpt", "gemini", "groq"]
                    tier = provider_health.get_best_available_provider(
                        chain, extension_connected=extension_connected
                    )
                    rationale = cls._format_frontend_rationale(tier)
                elif cls._FAST_API.search(clean):
                    chain = ["groq", "gemini", "claude"]
                    tier = provider_health.get_best_available_provider(
                        chain, extension_connected=extension_connected
                    )
                    rationale = f"[aivco-kernel] Analytical/coding query detected. Routing to Tier 2 ({tier.upper()})..."
                elif LIVE_SEARCH_PATTERN.search(clean):
                    chain = ["gemini", "chatgpt", "claude", "groq"]
                    tier = provider_health.get_best_available_provider(
                        chain, extension_connected=extension_connected
                    )
                    rationale = f"[aivco-kernel] Real-time/temporal query detected. Routing to web-grounded tier ({tier.upper()})..."
                else:
                    scores = compute_vector_scores(prompt)
                    if scores["context_need"] >= 0.8 and not DIRECTIVE_TRIGGER_CHECK.search(prompt):
                        tier = "kernel"
                        rationale = ""
                    elif scores["ui_density"] >= 0.7:
                        chain = ["claude", "chatgpt", "gemini", "groq"]
                        tier = provider_health.get_best_available_provider(
                            chain, extension_connected=extension_connected
                        )
                        rationale = cls._format_frontend_rationale(tier)
                    elif scores["logic_depth"] >= 0.7:
                        chain = ["groq", "gemini", "claude"]
                        tier = provider_health.get_best_available_provider(
                            chain, extension_connected=extension_connected
                        )
                        rationale = f"[aivco-kernel] Analytical/coding query detected. Routing to Tier 2 ({tier.upper()})..."
                    else:
                        chain = ["groq", "gemini"]
                        tier = provider_health.get_best_available_provider(
                            chain, extension_connected=extension_connected
                        )
                        rationale = f"[aivco-kernel] General query detected. Routing to Tier 2 ({tier.upper()})..."

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if log_telemetry:
            log_arbiter_telemetry(prompt, tier, rationale, latency_ms)

        return tier, rationale

    @staticmethod
    def _format_frontend_rationale(tier: str) -> str:
        if tier == "claude":
            return "[aivco-kernel] Heavy frontend/application query detected. Escalating to Brain 4 (Claude Web Bridge)..."
        if tier == "chatgpt":
            return "[aivco-kernel] Heavy frontend query detected. Claude on cooldown/unavailable; routing to Brain 4 (ChatGPT Web Bridge)..."
        if tier == "gemini":
            return "[aivco-kernel] Heavy frontend query detected. Web bridge on cooldown/unavailable; routing to Tier 3 (Gemini Pro API)..."
        return "[aivco-kernel] Heavy frontend query detected. Cascading to Tier 2 (Groq Fast API)..."

    @classmethod
    def flush_telemetry(cls, timeout: float = 2.0) -> None:
        flush_telemetry(timeout=timeout)


# ── Pre-Delivery Code Quality Gate & Brace-Balanced TypeScript Sanitizer ─────

def _strip_balanced_ts_blocks(code: str, keyword: str) -> str:
    """
    Safely removes `interface Foo { ... }` or `type Foo = { ... };` blocks
    with arbitrary nested `{ ... }` braces so inner braces never leave syntax errors.
    """
    pattern = re.compile(
        rf"\b{keyword}\s+[A-Za-z0-9_$]+(?:\s*<[^>]*>)?(?:\s+extends\s+[^{{=]+)?\s*(?:=\s*)?\{{",
        re.MULTILINE,
    )
    out: List[str] = []
    idx = 0
    n = len(code)
    while idx < n:
        m = pattern.search(code, idx)
        if not m:
            out.append(code[idx:])
            break
        out.append(code[idx : m.start()])
        # Walk balanced braces starting at the opening '{'
        brace_start = m.end() - 1
        depth = 0
        pos = brace_start
        in_str: Optional[str] = None
        escaped = False
        while pos < n:
            ch = code[pos]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif in_str:
                if ch == in_str:
                    in_str = None
            elif ch in ("'", '"', "`"):
                in_str = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    pos += 1
                    break
            pos += 1
        # Consume optional trailing semicolon and whitespace
        while pos < n and code[pos] in (" ", "\t", ";"):
            pos += 1
        idx = pos
    return "".join(out)


def _strip_param_types_in_list(raw_params: str) -> str:
    """
    Splits a parameter list at top-level commas (respecting nested parens/brackets/braces)
    and strips top-level `: Type` annotations while preserving default values (`= val`).
    """
    if ":" not in raw_params:
        return raw_params

    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in raw_params:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))

    cleaned_parts: List[str] = []
    for part in parts:
        p = part.strip()
        if not p:
            continue
        # Find top-level ':' and '='
        colon_idx = -1
        eq_idx = -1
        d = 0
        for i, ch in enumerate(p):
            if ch in "([{<":
                d += 1
            elif ch in ")]}>":
                d = max(0, d - 1)
            elif d == 0:
                if ch == ":" and colon_idx == -1 and eq_idx == -1:
                    colon_idx = i
                elif ch == "=" and eq_idx == -1:
                    eq_idx = i
                    break
        if colon_idx != -1:
            name_part = p[:colon_idx].rstrip("?").strip()
            if eq_idx != -1:
                default_val = p[eq_idx + 1 :].strip()
                cleaned_parts.append(f"{name_part} = {default_val}")
            else:
                cleaned_parts.append(name_part)
        else:
            cleaned_parts.append(p)

    return ", ".join(cleaned_parts)


def sanitize_script_content(js_code: str) -> str:
    """
    Sanitizes inline JavaScript by stripping accidental TypeScript syntax:
    - nested interface / type declarations (brace-balanced)
    - single-line type aliases (`type ID = string | number;`)
    - type assertions (`as HTMLElement`, `as unknown as Foo`, `as any`)
    - generic call arguments (`querySelector<HTMLElement>(...)`)
    - non-null assertions (`el!.addEventListener`)
    - variable, function parameter, and return type annotations
    """
    if not js_code or not isinstance(js_code, str):
        return js_code or ""

    # 1. Strip brace-balanced interface and object type declarations
    js_code = _strip_balanced_ts_blocks(js_code, "interface")
    js_code = _strip_balanced_ts_blocks(js_code, "type")

    # 2. Strip single-line union/primitive type aliases: type Mode = 'dark' | 'light';
    js_code = re.sub(
        r"^\s*(?:export\s+)?type\s+[A-Za-z0-9_$]+(?:\s*<[^>]*>)?\s*=\s*[^;\n]+;?[ \t]*$",
        "",
        js_code,
        flags=re.MULTILINE,
    )

    # 3. Strip generic type arguments on DOM/function calls: document.querySelector<HTMLElement>('...')
    js_code = re.sub(
        r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*<\s*[A-Za-z_$][A-Za-z0-9_$<>\[\],\s|&]*>\s*\(",
        r"\1(",
        js_code,
    )

    # 4. Strip type assertions (handles chained `as unknown as HTMLElement`)
    for _ in range(2):
        js_code = re.sub(
            r"\s+as\s+(?:const|[A-Za-z_$][A-Za-z0-9_$.]*(?:<[^>]+>)?(?:\[\])*)",
            "",
            js_code,
        )

    # 5. Strip non-null assertions: `element!.addEventListener`, `container!.innerHTML`, `el!;`
    js_code = re.sub(r"([a-zA-Z0-9_$\]\)])!([.\[(])", r"\1\2", js_code)
    js_code = re.sub(r"([a-zA-Z0-9_$\]\)])!\s*(?=[;,)\]\n])", r"\1", js_code)

    # 6. Strip variable type annotations on a single line: const x: number = ..., let data: Item[] = ...
    js_code = re.sub(
        r"\b(const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*[A-Za-z_$][A-Za-z0-9_$<>[\]|&,\s]*?(?=\s*=)",
        r"\1 \2",
        js_code,
    )

    # 7. Strip function parameter types and return types: function foo(a: number, b: string = 'x'): void {
    def _clean_fn_decl(m: re.Match) -> str:
        fn_name = m.group(1) or ""
        params = _strip_param_types_in_list(m.group(2) or "")
        space = f" {fn_name}" if fn_name else ""
        return f"function{space}({params}) {{"

    js_code = re.sub(
        r"\bfunction\s*([A-Za-z_$][A-Za-z0-9_$]*)?\s*\(([^()]*)\)\s*(?::\s*[A-Za-z_$][A-Za-z0-9_$<>[\]|&?\s]*)?\s*\{",
        _clean_fn_decl,
        js_code,
    )

    # 8. Strip arrow function parameter types and return types: (a: number, b: string): number =>
    def _clean_arrow_decl(m: re.Match) -> str:
        params = _strip_param_types_in_list(m.group(1) or "")
        return f"({params}) =>"

    js_code = re.sub(
        r"\(([^()\n]*:[^()\n]*)\)\s*(?::\s*[A-Za-z_$][A-Za-z0-9_$<>[\]|&?\s]*)?\s*=>",
        _clean_arrow_decl,
        js_code,
    )

    return js_code.strip()


def normalize_text_and_whitespace(text: str) -> str:
    """
    Normalizes prose tokens and fixes spacing artifacts:
    1. Preserves fenced code blocks (```...```) and inline code (`...`) verbatim.
    2. Fixes unspaced sentence collisions (.Sir, -> . Sir,).
    3. Fixes merged/smashed dates, numbers, and proper nouns:
       - 'AtléticoMadrid' -> 'Atlético Madrid'
       - 'FCBarcelona' -> 'FC Barcelona'
       - 'RealMadrid' -> 'Real Madrid'
       - '20May2026' -> '20 May 2026'
       - 'June2024' -> 'June 2024'
    4. Deduplicates consecutive identical lines/paragraphs caused by buffer re-stitching.
    """
    if not text or not isinstance(text, str):
        return text or ""

    code_blocks = {}
    counter = 0

    def code_repl(m):
        nonlocal counter
        token = f"___AIVCO_CODE_BLOCK_{counter}___"
        code_blocks[token] = m.group(0)
        counter += 1
        return token

    # Protect code blocks
    prose = re.sub(r"```[\s\S]*?```", code_repl, text)
    prose = re.sub(r"`[^`\n]+`", code_repl, prose)

    # 1. Unspaced period before capitalized word (.Sir, -> . Sir,)
    prose = re.sub(r"\.([A-Z][a-z]+)", r". \1", prose)

    # 2. Date normalization
    prose = re.sub(
        r"\b(\d{1,2})(January|February|March|April|May|June|July|August|September|October|November|December)(\d{4})\b",
        r"\1 \2 \3",
        prose,
        flags=re.IGNORECASE,
    )
    prose = re.sub(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)(\d{4})\b",
        r"\1 \2",
        prose,
        flags=re.IGNORECASE,
    )
    prose = re.sub(
        r"\b(\d{1,2})(January|February|March|April|May|June|July|August|September|October|November|December)\b",
        r"\1 \2",
        prose,
        flags=re.IGNORECASE,
    )
    prose = re.sub(r"\b(\d{1,4})([A-Z][a-z]+)\b", r"\1 \2", prose)

    # 3. Known entity smashes
    prose = re.sub(r"\bFCBarcelona\b", "FC Barcelona", prose)
    prose = re.sub(r"\bAtl[eé]ticoMadrid\b", "Atlético Madrid", prose, flags=re.IGNORECASE)
    prose = re.sub(r"\bRealMadrid\b", "Real Madrid", prose)
    prose = re.sub(r"\b([A-Z]{2,})([A-Z][a-z]{2,})\b", r"\1 \2", prose)

    # 4. Smashed unicode / CamelCase words in prose (lowercase followed by uppercase)
    prose = re.sub(r"([a-z\u00DF-\u00FF])([A-Z\u00C0-\u00DE])", r"\1 \2", prose)

    # 5. Deduplicate identical consecutive paragraphs
    paragraphs = prose.split("\n\n")
    deduped = []
    for p in paragraphs:
        p_strip = p.strip()
        if not p_strip:
            continue
        if deduped and p_strip == deduped[-1].strip():
            continue
        deduped.append(p)
    prose = "\n\n".join(deduped)

    # 6. Deduplicate repeated back-to-back sentences within lines
    prose = re.sub(r"([^.!?\n]{10,}[.!?])\s*\1", r"\1", prose)

    # Restore code blocks
    for token, block in code_blocks.items():
        prose = prose.replace(token, block)

    return prose.strip()


def sanitize_and_validate_code_output(markdown_text: str) -> str:
    """
    Pre-delivery quality gate that strips accidental TypeScript syntax out of plain <script> blocks
    (while leaving JSON/importmap/babel scripts untouched), balances HTML & markdown fences,
    and normalizes prose whitespace and deduplicates fallback duplicates.
    """
    if not markdown_text or not isinstance(markdown_text, str):
        return markdown_text or ""

    def _clean_script_tag(match: re.Match) -> str:
        open_tag = match.group(1)
        body = match.group(2)
        close_tag = match.group(3)
        open_lower = open_tag.lower()
        # Do not touch non-JS script blocks (e.g., application/json, importmap, ld+json, text/babel)
        if "type=" in open_lower and not any(
            js_t in open_lower
            for js_t in ("text/javascript", "application/javascript", "module")
        ):
            return match.group(0)
        return f"{open_tag}\n{sanitize_script_content(body)}\n{close_tag}"

    sanitized = re.sub(
        r"(<script(?:\s+[^>]*)?>)([\s\S]*?)(</script>)",
        _clean_script_tag,
        markdown_text,
        flags=re.IGNORECASE,
    )

    if "```html" in sanitized.lower():
        lower_san = sanitized.lower()
        if "<html" in lower_san and "</html>" not in lower_san:
            last_fence = sanitized.rfind("```")
            if last_fence > lower_san.find("```html"):
                sanitized = (
                    sanitized[:last_fence] + "\n</html>\n" + sanitized[last_fence:]
                )
            else:
                sanitized += "\n</html>\n```"
        else:
            if sanitized.count("```") % 2 != 0:
                sanitized += "\n```"

    sanitized = normalize_text_and_whitespace(sanitized)
    return sanitized