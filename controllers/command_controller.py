"""
AIVCO Command Controller

Decides whether a line of input is an AIVCO command (remember,
recall, forget, help, clear, exit, memory, core, project, episode, context)
or a normal question that should go to the LLM. Every command is handled
directly in Python -- never via subprocess -- and malformed commands are
always caught here, so they can never leak through to the model or the CLI.
"""

from typing import Optional

from memory.memory_controller import (
    DEFAULT_CATEGORY,
    DEFAULT_IMPORTANCE,
    MemoryController,
    VALID_CATEGORIES,
    VALID_IMPORTANCE,
)
from memory.multi_layer_memory import MultiLayerMemory
from memory.dataset_logger import FineTuneDatasetLogger

HELP_TEXT = """AIVCO commands:
  remember <key> = <value> [category=<category>] [importance=<importance>]   Save a memory
  recall <key>                Show one memory
  recall category <category>  Show memories in a category
  recall important <level>    Show memories by importance
  memory                      Alias for 'recall'
  memory search <query>       Search memories by key, value, or category
  memory stats                Show memory statistics
  forget <key>                Delete one memory
  forget all                  Delete ALL memories (asks to confirm)

  core set <key> = <value>            Save a Layer 1 core fact
  core get <key>                      Retrieve a Layer 1 core fact
  core delete <key>                   Delete a Layer 1 core fact

  project register <project_id> <title>   Register a project and set as active
  project use <project_id>                Switch active project for context
  project list                            List all registered projects
  project status [project_id]             Show project details and tasks
  project task add <project_id> <task>    Add a pending task to a project
  project task done <project_id> <task>   Mark a task as completed

  episode add <event> | <outcome>     Log an episodic memory entry
  episode list                        List recent episodic memory entries
  context show                        Print assembled V2 memory context

  teach <trigger> => <fact>           Teach AIVCO a permanent zero-LLM rule
  kernel status                       Show Cognitive Kernel operational status

  dataset stats                       Show fine-tune dataset entry count/sources
  dataset export                      Print path to the fine-tune dataset file

  clear                       Clear the screen
  help                        Show this message
  exit                        Quit AIVCO

Anything else is sent to the AI model.
Prefix with 'deep: ' for Gemini Tier 3, or 'private: ' for local-only mode."""

EXIT_SENTINEL = "__EXIT__"
CLEAR_SENTINEL = "__CLEAR__"


class CommandController:
    def __init__(
        self,
        memory: MemoryController,
        confirm_input=input,
        tiered_memory: Optional[MultiLayerMemory] = None,
        active_project_id: Optional[str] = None,
    ):
        self.memory = memory
        # Injectable so it can be swapped out in tests.
        self._confirm_input = confirm_input
        # V2 tiered memory — lazily initialize with defaults if not injected
        self.tiered_memory = tiered_memory or MultiLayerMemory(
            storage_path="data/v2_memory.json"
        )
        if active_project_id is not None:
            self.active_project_id: Optional[str] = active_project_id
        else:
            self.active_project_id = self.tiered_memory.get_active_project()
        # Subsystem 4: Fine-tune dataset logger
        self._dataset_logger = FineTuneDatasetLogger()

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
            if not remainder:
                return "recall_all"
            sub = remainder.split(None, 1)
            if sub[0].lower() == "search" and len(sub) == 2:
                return "memory_search"
            if sub[0].lower() == "stats" and len(sub) == 1:
                return "memory_stats"
            if sub[0].lower() in {"category", "importance", "important"} and len(sub) == 2:
                return "memory_filter"
            return "memory_bad"
        if keyword == "recall":
            if not remainder:
                return "recall_bad"
            tokens = remainder.split()
            first = tokens[0].lower()
            if first == "search":
                if len(tokens) < 2:
                    return "recall_bad"
                return "recall_search"
            if first in {"category", "importance", "important"}:
                if len(tokens) != 2:
                    return "recall_bad"
                return "recall_filter"
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

        # V2 commands
        if keyword == "core":
            if not remainder:
                return "core_bad"
            sub = remainder.split(None, 1)
            sub_cmd = sub[0].lower()
            if sub_cmd == "set" and len(sub) == 2 and "=" in sub[1]:
                return "core_set"
            if sub_cmd == "get" and len(sub) == 2:
                return "core_get"
            if sub_cmd == "delete" and len(sub) == 2:
                return "core_delete"
            return "core_bad"

        if keyword == "project":
            if not remainder:
                return "project_bad"
            sub = remainder.split(None, 1)
            sub_cmd = sub[0].lower()
            if sub_cmd == "register" and len(sub) == 2:
                parts = sub[1].split(None, 1)
                if len(parts) >= 1:
                    return "project_register"
                return "project_bad"
            if sub_cmd == "use" and len(sub) == 2:
                return "project_use"
            if sub_cmd == "list" and len(sub) == 1:
                return "project_list"
            if sub_cmd == "status":
                return "project_status"
            if sub_cmd == "task" and len(sub) == 2:
                task_sub = sub[1].split(None, 1)
                if len(task_sub) == 2 and task_sub[0].lower() in ("add", "done"):
                    return "project_task"
                return "project_bad"
            return "project_bad"

        if keyword == "episode":
            if not remainder:
                return "episode_bad"
            sub = remainder.split(None, 1)
            if sub[0].lower() == "add" and len(sub) == 2:
                return "episode_add"
            if sub[0].lower() == "list":
                return "episode_list"
            return "episode_bad"

        if keyword == "context":
            sub = remainder.split(None, 1)
            if not sub or sub[0].lower() == "show":
                return "context_show"
            return "context_bad"

        if keyword == "dataset":
            if not remainder:
                return "dataset_bad"
            sub = remainder.split(None, 1)
            if sub[0].lower() == "stats":
                return "dataset_stats"
            if sub[0].lower() == "export":
                return "dataset_export"
            return "dataset_bad"

        if keyword == "teach":
            if not remainder or "=>" not in remainder:
                return "teach_bad"
            return "teach"

        if keyword == "kernel":
            if not remainder or remainder.strip().lower() == "status":
                return "kernel_status"
            return "kernel_bad"

        return None

    def is_command(self, raw_input: str) -> bool:
        return self._classify(raw_input.strip().strip("`").strip()) is not None

    def _format_memory_line(self, key: str, record: dict) -> str:
        return (
            f"{key} = {record['value']} "
            f"[category={record['category']} importance={record['importance']}]"
        )

    def _format_memory_listing(self, records: dict) -> str:
        if not records:
            return "No memories stored."
        lines = ["===== AIVCO MEMORY ====="]
        lines += [self._format_memory_line(k, v) for k, v in records.items()]
        lines.append("=========================")
        return "\n".join(lines)

    def _parse_remember_options(self, remainder: str) -> tuple[str, str, str, str]:
        key, rest = remainder.split("=", 1)
        metadata = {
            "category": DEFAULT_CATEGORY,
            "importance": DEFAULT_IMPORTANCE,
        }
        value = rest.strip()
        tokens = value.split()
        # Support metadata tokens optionally wrapped in brackets, e.g.
        # [category=project] [importance=high]
        while tokens:
            last = tokens[-1]
            stripped = last
            if stripped.startswith("[") and stripped.endswith("]"):
                stripped = stripped[1:-1]
            if "=" not in stripped:
                break
            opt_key, opt_value = stripped.split("=", 1)
            opt_key = opt_key.strip().lower()
            opt_value = opt_value.strip().lower()
            if opt_key in metadata and opt_value:
                metadata[opt_key] = opt_value
                tokens.pop()
                value = " ".join(tokens).strip()
            else:
                break
        return key.strip(), value.strip(), metadata["category"], metadata["importance"]

    def handle(self, raw_input: str) -> Optional[str]:
        """
        Returns:
          - None            -> not a command, send raw_input to the LLM
          - ""               -> command handled, nothing to print (e.g. blank line)
          - "__EXIT__"       -> caller should shut down
          - "__CLEAR__"      -> caller should clear the screen
          - any other string -> command handled, print this message
        """
        text = raw_input.strip().strip("`").strip()
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
            return self._format_memory_listing(self.memory.all_records())

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

        if kind in {"recall_filter", "memory_filter"}:
            tokens = remainder.split(None, 1)
            filter_type = tokens[0].lower()
            filter_value = tokens[1].strip().lower() if len(tokens) > 1 else ""
            if filter_type == "category":
                if not self.memory.is_valid_category(filter_value):
                    return "Invalid category. Use one of: " + ", ".join(sorted(VALID_CATEGORIES))
                records = {
                    k: v
                    for k, v in self.memory.all_records().items()
                    if v["category"] == filter_value
                }
                if not records:
                    return f"No memories found in category: {filter_value}"
                return self._format_memory_listing(records)
            if filter_type in {"importance", "important"}:
                if not self.memory.is_valid_importance(filter_value):
                    return "Invalid importance. Use one of: " + ", ".join(sorted(VALID_IMPORTANCE))
                records = {
                    k: v
                    for k, v in self.memory.all_records().items()
                    if v["importance"] == filter_value
                }
                if not records:
                    return f"No memories found with importance: {filter_value}"
                return self._format_memory_listing(records)
            return "Usage: recall category <category> or recall important <level>"

        if kind in {"recall_search", "memory_search"}:
            query = remainder.split(None, 1)[1] if len(remainder.split(None, 1)) > 1 else ""
            if not query:
                return "Usage: memory search <query>"
            results = self.memory.search(query)
            if not results:
                return "No matching memories found."
            return self._format_memory_listing(results)

        if kind == "memory_bad":
            return "Usage: memory search <query>"

        if kind == "memory_stats":
            summary = self.memory.stats()
            lines = [
                f"Total memories: {summary['total']}",
                "Importance:",
            ]
            lines += [f"  {level}: {count}" for level, count in summary["importance"].items()]
            if summary["categories"]:
                lines.append("Categories:")
                lines += [f"  {category}: {count}" for category, count in summary["categories"].items()]
            return "\n".join(lines)

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
            key, value, category, importance = self._parse_remember_options(remainder)
            if not key or not value:
                return "Usage: remember <key> = <value>"
            if not self.memory.is_valid_key(key):
                return "Invalid key. Use only letters, numbers, '_' or '-' (max 64 characters)."
            _, msg = self.memory.remember(key, value, category=category, importance=importance)
            return msg

        # ── V2 Core Profile Commands ──────────────────────────────────────────

        if kind == "core_bad":
            return "Usage: core set <key> = <value> | core get <key> | core delete <key>"

        if kind == "core_set":
            # remainder = "set <key> = <value>"
            rest_after_set = remainder.split(None, 1)[1]  # drop "set"
            key_part, val_part = rest_after_set.split("=", 1)
            key = key_part.strip()
            value = val_part.strip()
            if not key or not value:
                return "Usage: core set <key> = <value>"
            self.tiered_memory.set_core_fact(key, value)
            return f"[Core] Saved: {key} = {value}"

        if kind == "core_get":
            key = remainder.split(None, 1)[1].strip()
            value = self.tiered_memory.get_core_fact(key)
            if value is None:
                return f"[Core] Not found: {key}"
            return f"[Core] {key} = {value}"

        if kind == "core_delete":
            key = remainder.split(None, 1)[1].strip()
            deleted = self.tiered_memory.delete_core_fact(key)
            if deleted:
                return f"[Core] Deleted: {key}"
            return f"[Core] Not found: {key}"

        # ── V2 Project Commands ───────────────────────────────────────────────

        if kind == "project_bad":
            return (
                "Usage: project register <id> <title> | project use <id> | "
                "project list | project status [id] | "
                "project task add <id> <task> | project task done <id> <task>"
            )

        if kind == "project_register":
            # remainder = "register <project_id> <title>"
            rest = remainder.split(None, 1)[1]  # drop "register"
            parts = rest.split(None, 1)
            project_id = parts[0].strip()
            title = parts[1].strip() if len(parts) > 1 else project_id
            self.tiered_memory.register_project(project_id, title)
            self.active_project_id = project_id
            return f"[Project] Registered '{title}' ({project_id}). Active project set to: {project_id}"

        if kind == "project_use":
            project_id = remainder.split(None, 1)[1].strip()
            if project_id not in self.tiered_memory.projects:
                return f"[Project] Error: Project '{project_id}' not found. (Unknown project)"
            self.tiered_memory.set_active_project(project_id)
            self.active_project_id = project_id
            return f"[Project] Active project switched to: {project_id}"

        if kind == "project_list":
            if not self.tiered_memory.projects:
                return "[Project] No registered projects found. Use 'project register <id> <title>' to create one."
            lines = ["===== REGISTERED PROJECTS ====="]
            current_active = self.active_project_id or self.tiered_memory.get_active_project()
            for pid, proj in sorted(self.tiered_memory.projects.items()):
                is_active = (pid == current_active)
                marker = "* " if is_active else "  "
                active_str = " (Active)" if is_active else ""
                lines.append(
                    f"{marker}{pid}{active_str} - {proj.title} "
                    f"[{len(proj.completed_milestones)} milestones, {len(proj.pending_tasks)} pending]"
                )
            lines.append("===============================")
            return "\n".join(lines)

        if kind == "project_status":
            tokens = remainder.split(None, 1)
            target_id = tokens[1].strip() if len(tokens) > 1 else (self.active_project_id or self.tiered_memory.get_active_project())
            if not target_id or target_id not in self.tiered_memory.projects:
                return "[Project] No active project set. Specify one or use 'project use <id>'."
            proj = self.tiered_memory.projects[target_id]
            stack_str = ", ".join(proj.tech_stack) if proj.tech_stack else "None specified"
            lines = [
                f"Project: {proj.title} ({proj.project_id}) [Status: {proj.status}]",
                f"Tech Stack: {stack_str}",
                f"Completed Milestones ({len(proj.completed_milestones)}):",
            ]
            if proj.completed_milestones:
                for m in proj.completed_milestones:
                    lines.append(f"  - {m}")
            else:
                lines.append("  (None)")
            lines.append(f"Pending Tasks ({len(proj.pending_tasks)}):")
            if proj.pending_tasks:
                for t in proj.pending_tasks:
                    lines.append(f"  - {t}")
            else:
                lines.append("  (None)")
            return "\n".join(lines)

        if kind == "project_task":
            # remainder = "task add|done <project_id> <task>"
            rest = remainder.split(None, 1)[1]  # drop "task"
            task_parts = rest.split(None, 1)
            action = task_parts[0].lower()
            id_and_task = task_parts[1].split(None, 1) if len(task_parts) > 1 else []
            if len(id_and_task) < 2:
                return "Usage: project task add|done <project_id> <task>"
            project_id = id_and_task[0].strip()
            task = id_and_task[1].strip()
            mark_done = (action == "done")
            try:
                self.tiered_memory.update_project_task(project_id, task, mark_completed=mark_done)
            except KeyError:
                return f"[Project] Unknown project: {project_id}"
            verb = "completed" if mark_done else "added"
            return f"[Project] Task {verb}: {task}"

        # ── V2 Episode Commands ───────────────────────────────────────────────

        if kind == "episode_bad":
            return "Usage: episode add <event> | <outcome> | episode list"

        if kind == "episode_list":
            eps = self.tiered_memory.get_recent_episodes(limit=10, project_id=self.active_project_id)
            if not eps:
                return "[Episode] No episodes logged yet."
            lines = ["===== RECENT EPISODES ====="]
            for ep in eps:
                date = ep.timestamp[:10] if ep.timestamp else "recent"
                proj = f" [{ep.project_id}]" if ep.project_id else ""
                lines.append(f"  [{date}]{proj} {ep.event} | {ep.outcome}")
            lines.append("===========================")
            return "\n".join(lines)

        if kind == "episode_add":
            # remainder = "add <event> [| <outcome>]"
            rest = remainder.split(None, 1)[1]  # drop "add"
            if "|" in rest:
                event_part, outcome_part = rest.split("|", 1)
                event = event_part.strip()
                outcome = outcome_part.strip() or "logged"
            else:
                event = rest.strip()
                outcome = "logged"
            if not event:
                return "Usage: episode add <event> | <outcome>"
            ep = self.tiered_memory.add_episode(
                event=event,
                outcome=outcome,
                project_id=self.active_project_id,
            )
            return f"[Episode] Logged [{ep.id}]: {event}"

        # ── V2 Context Commands ───────────────────────────────────────────────

        if kind in ("context_show", "context_bad"):
            ctx = self.tiered_memory.assemble_context(
                active_project_id=self.active_project_id
            )
            if not ctx:
                return "[Context] No V2 context available yet."
            return ctx

        # ── Dataset Commands (Subsystem 4) ────────────────────────────────────

        if kind == "dataset_stats":
            return self._dataset_logger.format_stats_report()

        if kind == "dataset_export":
            return f"[Dataset] Export path: {self._dataset_logger.export_path()}"

        if kind == "dataset_bad":
            return "Usage: dataset stats | dataset export"

        # ── Teach & Kernel Commands (Cognitive Kernel) ────────────────────────

        if kind == "teach_bad":
            return "Usage: teach <trigger> => <fact>"

        if kind == "teach":
            parts = remainder.split("=>", 1)
            trigger = parts[0].strip()
            fact = parts[1].strip()
            if not trigger or not fact:
                return "Usage: teach <trigger> => <fact>"

            if not hasattr(self.tiered_memory, "learned_facts"):
                self.tiered_memory.learned_facts = []

            import re
            keywords = [k for k in re.findall(r"\b\w+\b", trigger.lower()) if len(k) >= 3]
            found = False
            for entry in self.tiered_memory.learned_facts:
                if entry.get("trigger", "").strip().lower() == trigger.lower() or entry.get("fact", "").strip().lower() == fact.lower():
                    entry["trigger"] = trigger
                    entry["trigger_keywords"] = keywords
                    entry["fact"] = fact
                    entry["hit_count"] = entry.get("hit_count", 0) + 1
                    found = True
                    break
            if not found:
                from datetime import datetime, timezone
                self.tiered_memory.learned_facts.append({
                    "trigger": trigger,
                    "trigger_keywords": keywords,
                    "fact": fact,
                    "fact_type": "taught_rule",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "hit_count": 0,
                })
            self.tiered_memory.save()
            pref = getattr(self.tiered_memory, "get_core_fact", lambda k: "Sir")("preferred_address") or "Sir"
            return f"[aivco-kernel] Learned permanent rule, {pref}: '{trigger}' -> '{fact}'"

        if kind == "kernel_status":
            num_learned = len(getattr(self.tiered_memory, "learned_facts", []))
            return (
                "===== AIVCO COGNITIVE KERNEL STATUS =====\n"
                "Stage 0/1 Zero-LLM Resolver : ACTIVE\n"
                f"Learned Rules/Facts In-Memory: {num_learned}\n"
                "Subsystem 2 Continuous Learn  : ACTIVE\n"
                "Subsystem 3 Autonomous Arbiter: ACTIVE\n"
                "Subsystem 4 Dataset Logger    : ACTIVE\n"
                "=========================================="
            )

        if kind == "kernel_bad":
            return "Usage: kernel status"

        return None
