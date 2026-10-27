"""
AIVCO Fine-Tune Dataset Logger (Subsystem 4)

Logs every high-value interaction to a JSONL file for future
LoRA / MLX / Unsloth fine-tuning on Mac (ShareGPT / ChatML / Alpaca format).

Usage:
    from memory.dataset_logger import FineTuneDatasetLogger

    logger = FineTuneDatasetLogger()
    logger.log(
        source="groq",
        task_type="general",
        system_content="You are AIVCO...",
        user_content="Explain X",
        assistant_content="...",
    )

CLI commands (routed through CommandController):
    dataset export   -> prints the path to the current JSONL file
    dataset stats    -> prints count, sources, task types
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DATASET_PATH = "data/fine_tune_dataset.jsonl"


class FineTuneDatasetLogger:
    """
    Append-only JSONL logger for LLM interaction data.
    Each entry is a ShareGPT-compatible record with messages array.
    """

    def __init__(self, filepath: str = DEFAULT_DATASET_PATH):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        source: str,
        task_type: str,
        system_content: str,
        user_content: str,
        assistant_content: str,
        extra_meta: Optional[dict] = None,
    ) -> None:
        """
        Append a single interaction to the JSONL dataset file.

        Args:
            source:            Provider that generated the response
                               ("aivco_kernel" | "local_llama" | "groq" | "gemini")
            task_type:         Routing task type ("general" | "deep" | "private" etc.)
            system_content:    System prompt (AIVCO persona + memory context)
            user_content:      Original user prompt
            assistant_content: Final assistant response that was shown to user
            extra_meta:        Optional dict of extra metadata to include
        """
        if not assistant_content or not assistant_content.strip():
            return  # don't log empty responses

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "task_type": task_type,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        }
        if extra_meta:
            entry["meta"] = extra_meta

        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # never crash the main loop due to logging failure

    def stats(self) -> dict:
        """
        Return statistics about the dataset:
          total_entries, sources (counts), task_types (counts)
        """
        total = 0
        sources: dict[str, int] = {}
        task_types: dict[str, int] = {}

        if not self.filepath.exists():
            return {"total_entries": 0, "sources": {}, "task_types": {}}

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        total += 1
                        src = entry.get("source", "unknown")
                        tt = entry.get("task_type", "unknown")
                        sources[src] = sources.get(src, 0) + 1
                        task_types[tt] = task_types.get(tt, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

        return {
            "total_entries": total,
            "sources": sources,
            "task_types": task_types,
        }

    def export_path(self) -> str:
        """Return the absolute path to the dataset file."""
        return str(self.filepath.resolve())

    def format_stats_report(self) -> str:
        """Return a human-readable stats report string."""
        s = self.stats()
        lines = [
            f"Fine-Tune Dataset: {self.export_path()}",
            f"Total entries: {s['total_entries']}",
        ]
        if s["sources"]:
            lines.append("Sources:")
            for src, count in sorted(s["sources"].items()):
                lines.append(f"  {src}: {count}")
        if s["task_types"]:
            lines.append("Task types:")
            for tt, count in sorted(s["task_types"].items()):
                lines.append(f"  {tt}: {count}")
        return "\n".join(lines)

