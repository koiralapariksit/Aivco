"""
AIVCO Multi-Layer Memory Engine (Version 2.0)
Implements Layer 1 (Core Profile), Layer 2 (Episodic), and Layer 3 (Project Memory).
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
import os
import uuid


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CoreFact:
    key: str
    value: str
    importance: str = "high"
    updated_at: str = field(default_factory=_utc_iso_now)


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
    def __init__(self, storage_path: str = "data/multilayer_profile.json"):
        self.storage_path = storage_path
        self.core_profile: Dict[str, CoreFact] = {}
        self.episodes: List[Episode] = []
        self.projects: Dict[str, ProjectState] = {}
        self.load()

    # --- Layer 1: Core Profile ---
    def set_core_fact(self, key: str, value: str, importance: str = "high") -> CoreFact:
        fact = CoreFact(key=key, value=value, importance=importance)
        self.core_profile[key] = fact
        self.save()
        return fact

    def get_core_fact(self, key: str) -> Optional[str]:
        fact = self.core_profile.get(key)
        return fact.value if fact else None

    def delete_core_fact(self, key: str) -> bool:
        if key in self.core_profile:
            del self.core_profile[key]
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
        self.save()
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
        self, active_project_id: Optional[str] = None, max_episodes: int = 3
    ) -> str:
        sections = []

        if self.core_profile:
            core_lines = [f"- {fact.key}: {fact.value}" for fact in self.core_profile.values()]
            sections.append("### Core Profile Context\n" + "\n".join(core_lines))

        if active_project_id and active_project_id in self.projects:
            proj = self.projects[active_project_id]
            proj_info = [
                f"### Active Project: {proj.title} ({proj.project_id})",
                f"- Status: {proj.status}",
                f"- Stack: {', '.join(proj.tech_stack) if proj.tech_stack else 'N/A'}",
                f"- Completed: {len(proj.completed_milestones)} milestones",
                f"- Pending: {', '.join(proj.pending_tasks) if proj.pending_tasks else 'None'}",
            ]
            sections.append("\n".join(proj_info))

        recent_eps = self.get_recent_episodes(limit=max_episodes, project_id=active_project_id)
        if recent_eps:
            ep_lines = [
                f"- [{e.timestamp[:10]}] Event: {e.event} | Outcome: {e.outcome}"
                for e in recent_eps
            ]
            sections.append("### Recent History / Lessons Learned\n" + "\n".join(ep_lines))

        return "\n\n".join(sections)

    # --- Disk Storage ---
    def save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        data = {
            "version": "2.0-tiered",
            "core_profile": {k: asdict(v) for k, v in self.core_profile.items()},
            "episodes": [asdict(e) for e in self.episodes],
            "projects": {k: asdict(v) for k, v in self.projects.items()},
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return

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
        except Exception:
            pass
