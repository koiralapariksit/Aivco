"""
AIVCO Multi-Layer Memory Engine (Version 2.0)
Implements Layer 1 (Core Profile), Layer 2 (Episodic), and Layer 3 (Project Memory).
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager
import hashlib
import fcntl
import json
import os
import tempfile
import threading
import uuid
from memory.semantic_index import SemanticIndex


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_PATH_LOCKS: Dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


@dataclass
class CoreFact:
    key: str
    value: str
    importance: str = "high"
    updated_at: str = field(default_factory=_utc_iso_now)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        if isinstance(other, bool):
            return (self.value.lower() in ("true", "1", "yes")) if other else (self.value.lower() in ("false", "0", "no"))
        if isinstance(other, CoreFact):
            return self.key == other.key and self.value == other.value and self.importance == other.importance
        return False


@dataclass
class Episode:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event: str = ""
    outcome: str = ""
    timestamp: str = field(default_factory=_utc_iso_now)
    tags: List[str] = field(default_factory=list)
    project_id: Optional[str] = None


@dataclass
class ProjectState:
    project_id: str
    title: str
    status: str = "active"
    tech_stack: List[str] = field(default_factory=list)
    completed_milestones: List[str] = field(default_factory=list)
    pending_tasks: List[str] = field(default_factory=list)
    known_issues: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=_utc_iso_now)


class MultiLayerMemory:
    def __init__(self, storage_path: Optional[str] = None):
        data_dir = Path(os.environ.get("AIVCO_DATA_DIR", "./data"))
        self.storage_path = Path(storage_path) if storage_path is not None else data_dir / "multilayer_profile.json"
        self._thread_lock = _thread_lock_for(self.storage_path)
        self._deleted_core_keys = set()
        index_dir = Path(os.environ.get("AIVCO_DATA_DIR", str(self.storage_path.parent)))
        self.semantic_index = SemanticIndex(index_dir)
        self.core_profile: Dict[str, CoreFact] = {}
        self.episodes: List[Episode] = []
        self.projects: Dict[str, ProjectState] = {}
        self.active_project_id: Optional[str] = None
        # Subsystem 2: Learned facts cache (persisted in JSON)
        self.learned_facts: List[dict] = []
        self.load()
        # Auto-select single project if active_project_id is not set
        if not self.active_project_id and len(self.projects) == 1:
            self.active_project_id = next(iter(self.projects.keys()))
            self.save()

    @property
    def session_ledger(self):
        """Reference to the singleton SessionExecutionLedger."""
        from kernel.arbiter import session_ledger
        return session_ledger

    # --- Active Project Management ---
    def set_active_project(self, project_id: Optional[str]) -> None:
        if project_id is not None and project_id not in self.projects:
            raise KeyError(f"Project '{project_id}' not found.")
        self.active_project_id = project_id
        self.save()

    def _sync_semantic_index(self) -> None:
        source = hashlib.sha256(str(self.storage_path.resolve()).encode("utf-8")).hexdigest()[:16]
        prefix = f"v2:{source}:"
        entries = []
        for key, fact in self.core_profile.items():
            value = fact.value if isinstance(fact, CoreFact) else str(fact)
            updated = fact.updated_at if isinstance(fact, CoreFact) else _utc_iso_now()
            entries.append((f"fact:{key}", "core_profile", f"{key}: {value}", updated))
        for episode in self.episodes:
            entries.append((f"episode:{episode.id}", "episode", f"{episode.event} {episode.outcome}", episode.timestamp))
        for project_id, project in self.projects.items():
            content = " ".join((project.title, project.status, *project.tech_stack, *project.pending_tasks, *project.completed_milestones, *project.known_issues))
            entries.append((f"project:{project_id}", "project", content, project.last_updated))
        for index, item in enumerate(self.learned_facts):
            content = " ".join(str(item.get(k, "")) for k in ("trigger", "fact", "fact_type", "trigger_keywords"))
            entries.append((f"learned:{index}", "learned", content, item.get("timestamp", "")))
        self.semantic_index.sync_source(prefix, entries)

    def get_active_project(self) -> Optional[str]:
        return self.active_project_id

    # --- Layer 1: Core Profile ---
    def set_core_fact(self, key: str, value: str, importance: str = "high") -> CoreFact:
        fact = CoreFact(key=key, value=value, importance=importance)
        self.core_profile[key] = fact
        self._deleted_core_keys.discard(key)
        self.save()
        return fact

    def get_core_fact(self, key: str) -> Optional[str]:
        fact = self.core_profile.get(key)
        if fact is None:
            return None
        return fact.value if isinstance(fact, CoreFact) else str(fact)

    def delete_core_fact(self, key: str) -> bool:
        if key in self.core_profile:
            del self.core_profile[key]
            self._deleted_core_keys.add(key)
            self.save()
            return True
        return False

    # --- Layer 2: Episodic Memory ---
    def add_episode(
        self,
        event: str,
        outcome: str,
        tags: Optional[List[str]] = None,
        project_id: Optional[str] = None,
    ) -> Episode:
        ep = Episode(event=event, outcome=outcome, tags=tags or [], project_id=project_id)
        self.episodes.append(ep)
        self.save()
        return ep

    def get_recent_episodes(
        self, limit: int = 5, project_id: Optional[str] = None
    ) -> List[Episode]:
        filtered = [e for e in self.episodes if not project_id or e.project_id == project_id]
        return sorted(filtered, key=lambda x: x.timestamp, reverse=True)[:limit]

    # --- Layer 3: Project Memory ---
    def register_project(
        self, project_id: str, title: str, tech_stack: Optional[List[str]] = None
    ) -> ProjectState:
        if project_id in self.projects:
            proj = self.projects[project_id]
            proj.title = title
            if tech_stack is not None:
                proj.tech_stack = tech_stack
            proj.last_updated = _utc_iso_now()
        else:
            proj = ProjectState(
                project_id=project_id,
                title=title,
                tech_stack=tech_stack or [],
            )
            self.projects[project_id] = proj
        self.set_active_project(project_id)
        return proj

    def update_project_task(
        self, project_id: str, task: str, mark_completed: bool = False
    ) -> None:
        proj = self.projects.get(project_id)
        if not proj:
            raise KeyError(f"Project '{project_id}' not found.")

        if mark_completed:
            if task in proj.pending_tasks:
                proj.pending_tasks.remove(task)
            if task not in proj.completed_milestones:
                proj.completed_milestones.append(task)
        else:
            if task not in proj.pending_tasks:
                proj.pending_tasks.append(task)

        proj.last_updated = _utc_iso_now()
        self.save()

    # --- Layer 4: Context Assembly for the LLM ---
    def assemble_context(
        self,
        active_project_id: Optional[str] = None,
        max_episodes: int = 3,
        cloud_safe: bool = False,
    ) -> str:
        # If memory is completely empty, return empty string
        has_data = bool(self.core_profile or self.projects or self.episodes or getattr(self, "learned_facts", []))
        if not has_data:
            return ""

        if active_project_id is None:
            active_project_id = self.get_active_project()

        sections = []

        # High-Priority Rule Injection (Strict Operator Directives at the very top)
        directives = []
        preferred_addr = self.get_core_fact("preferred_address") or "Sir"
        directives.append(f"- Operator Address: Address the operator strictly as '{preferred_addr}' or 'AIvco'. Never address the operator by birth name unless asked.")

        rules_fact = self.get_core_fact("rules")
        if rules_fact:
            directives.append(f"- Operating Rules: {rules_fact}")

        concise_fact = self.get_core_fact("concise_mode")
        if concise_fact and str(concise_fact).lower() in ("true", "1", "yes"):
            directives.append("- Brevity Protocol: Keep casual chat and identity responses strictly concise (1-2 sentences). Elaborate only when deep technical code or architectural explanation is explicitly requested.")

        for lf in getattr(self, "learned_facts", []):
            if lf.get("fact_type") == "taught_rule" or "rule" in str(lf.get("fact_type", "")):
                directives.append(f"- Learned Rule: '{lf.get('trigger', '')}' -> {lf.get('fact', '')}")

        if directives:
            sections.append("### STRICT OPERATOR DIRECTIVES (HIGHEST PRIORITY):\n" + "\n".join(directives))

        if self.core_profile:
            core_lines = []
            for k, fact in self.core_profile.items():
                if isinstance(fact, CoreFact):
                    val = fact.value
                    imp = fact.importance
                else:
                    val = str(fact)
                    imp = "high"
                # If cloud_safe, scrub sensitive data or facts marked with [private]
                if cloud_safe:
                    if "[private]" in val.lower() or imp == "private":
                        continue
                    # Redact potential raw credentials or tokens
                    if any(s in k.lower() for s in ("token", "password", "secret", "api_key", "credential")):
                        continue
                core_lines.append(f"- {k}: {val}")
            if core_lines:
                sections.append("### Core Profile Context\n" + "\n".join(core_lines))

        if active_project_id and active_project_id in self.projects:
            proj = self.projects[active_project_id]
            pending_filtered = [t for t in proj.pending_tasks if not (cloud_safe and "[private]" in t.lower())]
            proj_info = [
                f"### Active Project: {proj.title} ({proj.project_id})",
                f"- Status: {proj.status}",
                f"- Stack: {', '.join(proj.tech_stack) if proj.tech_stack else 'N/A'}",
                f"- Completed: {len(proj.completed_milestones)} milestones",
                f"- Pending: {', '.join(pending_filtered) if pending_filtered else 'None'}",
            ]
            sections.append("\n".join(proj_info))

        recent_eps = self.get_recent_episodes(limit=max_episodes, project_id=active_project_id)
        if cloud_safe:
            recent_eps = [e for e in recent_eps if "[private]" not in e.event.lower()]
        if recent_eps:
            ep_lines = [
                f"- [{e.timestamp[:10]}] Event: {e.event} | Outcome: {e.outcome}"
                for e in recent_eps
            ]
            sections.append("### Recent History / Lessons Learned\n" + "\n".join(ep_lines))

        return "\n\n".join(sections)

    # --- Disk Storage ---
    @property
    def _lock_path(self) -> Path:
        return self.storage_path.with_name(self.storage_path.name + ".lock")

    @contextmanager
    def _locked_file(self, exclusive: bool):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            with self._lock_path.open("a+b") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read_json(path: Path):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None

    def _valid_backups(self) -> List[Path]:
        backup_dir = self.storage_path.parent / ".backups"
        if not backup_dir.exists():
            return []
        valid = []
        for path in backup_dir.glob(self.storage_path.name + ".*.json"):
            try:
                if self._read_json(path) is not None:
                    valid.append(path)
            except (OSError, ValueError):
                continue
        return sorted(valid, key=lambda p: p.stat().st_mtime_ns, reverse=True)

    def _restore_backup_locked(self) -> Optional[dict]:
        for backup in self._valid_backups():
            try:
                data = self._read_json(backup)
                if data is not None:
                    self._atomic_json_write(data)
                    return data
            except (OSError, ValueError):
                continue
        return None

    def _backup_current_locked(self) -> None:
        try:
            current = self._read_json(self.storage_path)
        except (OSError, ValueError):
            return
        if current is None:
            return
        backup_dir = self.storage_path.parent / ".backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = backup_dir / f"{self.storage_path.name}.{stamp}.{uuid.uuid4().hex[:8]}.json"
        backup_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        backups = self._valid_backups()
        for old in backups[5:]:
            try:
                old.unlink()
            except OSError:
                pass

    def _atomic_json_write(self, data: dict) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.storage_path.name + ".", suffix=".tmp", dir=self.storage_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            with open(tmp_name, "r", encoding="utf-8") as f:
                json.load(f)
            os.replace(tmp_name, self.storage_path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _merge_snapshots(disk: dict, current: dict) -> dict:
        """Preserve independently-added facts and episodes from stale instances."""
        merged = dict(disk)
        merged.update(current)
        for field_name, date_key in (("core_profile", "updated_at"), ("projects", "last_updated")):
            old_items = disk.get(field_name, {})
            new_items = current.get(field_name, {})
            if isinstance(old_items, dict) and isinstance(new_items, dict):
                combined = dict(old_items)
                for key, value in new_items.items():
                    old = combined.get(key)
                    if not isinstance(old, dict) or value.get(date_key, "") >= old.get(date_key, ""):
                        combined[key] = value
                merged[field_name] = combined
        old_eps, new_eps = disk.get("episodes", []), current.get("episodes", [])
        if isinstance(old_eps, list) and isinstance(new_eps, list):
            by_id = {e.get("id"): e for e in old_eps if isinstance(e, dict) and e.get("id")}
            by_id.update({e.get("id"): e for e in new_eps if isinstance(e, dict) and e.get("id")})
            merged["episodes"] = list(by_id.values())
        old_facts, new_facts = disk.get("learned_facts", []), current.get("learned_facts", [])
        if isinstance(old_facts, list) and isinstance(new_facts, list):
            merged["learned_facts"] = list({
                json.dumps(x, sort_keys=True): x for x in old_facts + new_facts
            }.values())
        return merged

    def save(self) -> None:
        core_dict = {}
        for k, v in self.core_profile.items():
            if isinstance(v, CoreFact):
                core_dict[k] = asdict(v)
            else:
                core_dict[k] = {
                    "key": k,
                    "value": str(v),
                    "importance": "high",
                    "updated_at": _utc_iso_now(),
                }
        data = {
            "version": "2.0-tiered",
            "active_project_id": self.active_project_id,
            "core_profile": core_dict,
            "episodes": [asdict(e) for e in self.episodes],
            "projects": {k: asdict(v) for k, v in self.projects.items()},
            "learned_facts": self.learned_facts,
        }
        with self._locked_file(exclusive=True):
            try:
                disk = self._read_json(self.storage_path) or {}
            except (OSError, ValueError):
                disk = {}
            data = self._merge_snapshots(disk, data)
            for key in self._deleted_core_keys:
                data.get("core_profile", {}).pop(key, None)
            self._backup_current_locked()
            self._atomic_json_write(data)
            self._deleted_core_keys.clear()
        self._sync_semantic_index()

    def load(self) -> None:
        with self._locked_file(exclusive=False):
            try:
                data = self._read_json(self.storage_path) if self.storage_path.exists() else None
            except (OSError, ValueError):
                data = None
        if data is None:
            with self._locked_file(exclusive=True):
                try:
                    data = self._read_json(self.storage_path) if self.storage_path.exists() else None
                except (OSError, ValueError):
                    data = None
                if data is None:
                    data = self._restore_backup_locked()
        if data is None:
            self._sync_semantic_index()
            return
        try:
            self.active_project_id = data.get("active_project_id")

            self.core_profile = {}
            core_dict = data.get("core_profile")
            if isinstance(core_dict, dict):
                for k, v in core_dict.items():
                    if isinstance(v, dict):
                        self.core_profile[k] = CoreFact(**v)

            self.episodes = []
            episodes_list = data.get("episodes")
            if isinstance(episodes_list, list):
                for e in episodes_list:
                    if isinstance(e, dict):
                        self.episodes.append(Episode(**e))

            self.projects = {}
            projects_dict = data.get("projects")
            if isinstance(projects_dict, dict):
                for k, v in projects_dict.items():
                    if isinstance(v, dict):
                        self.projects[k] = ProjectState(**v)

            # Load learned facts cache (Subsystem 2)
            self.learned_facts = []
            lf = data.get("learned_facts")
            if isinstance(lf, list):
                self.learned_facts = [e for e in lf if isinstance(e, dict)]

            # Validate loaded active_project_id exists in projects
            if self.active_project_id and self.active_project_id not in self.projects:
                self.active_project_id = None
            self._sync_semantic_index()
        except Exception:
            with self._locked_file(exclusive=True):
                restored = self._restore_backup_locked()
            if restored is not None:
                self.load()
