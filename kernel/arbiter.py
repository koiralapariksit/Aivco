"""
AIVCO Kernel Cognitive Arbiter & Executive Context Broker
Deterministically classifies incoming user queries into Kernel (Zero-LLM),
Brain 4 Web AI (Claude/ChatGPT via Extension Bridge), or Tier 2 Fast API (Groq/Gemini).
Runs in <5ms with 0-LLM token consumption.
Broker provides executive context envelopes and asynchronous telemetry training logging.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import threading
import time
from typing import Tuple, Optional, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TELEMETRY_FILE = BASE_DIR / "data" / "arbiter_training_log.jsonl"


# ── Asynchronous Telemetry Logger Queue & Daemon Worker ───────────────────────

_telemetry_queue: queue.Queue = queue.Queue(maxsize=50000)
_custom_telemetry_path: Optional[Path] = None


def _telemetry_worker():
    while True:
        try:
            item = _telemetry_queue.get(timeout=0.05)
            batch = [item]
            while not _telemetry_queue.empty() and len(batch) < 200:
                try:
                    batch.append(_telemetry_queue.get_nowait())
                except queue.Empty:
                    break

            grouped = {}
            for entry, target_path in batch:
                grouped.setdefault(target_path, []).append(entry)

            for target_path, entries in grouped.items():
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(target_path, "a", encoding="utf-8") as f:
                        for entry in entries:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            for _ in batch:
                _telemetry_queue.task_done()
        except queue.Empty:
            continue
        except Exception:
            pass


_worker_thread = threading.Thread(target=_telemetry_worker, daemon=True, name="ArbiterTelemetryWorker")
_worker_thread.start()


def set_telemetry_filepath(filepath: Optional[str]) -> None:
    """Sets a custom filepath for telemetry logging (useful for unit testing)."""
    global _custom_telemetry_path
    _custom_telemetry_path = Path(filepath) if filepath else None


def flush_telemetry(timeout: float = 2.0) -> None:
    """Drains the telemetry queue up to timeout seconds."""
    try:
        _telemetry_queue.join()
    except Exception:
        pass


def log_arbiter_telemetry(
    prompt: str,
    target_tier: str,
    rationale: str,
    latency_ms: float,
    filepath: Optional[str] = None,
) -> None:
    """Non-blocking asynchronous append to arbiter_training_log.jsonl."""
    target_path = Path(filepath) if filepath else (_custom_telemetry_path or DEFAULT_TELEMETRY_FILE)
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
        pass


# ── Executive Context Broker Helpers ─────────────────────────────────────────

def compress_memory_context(memory_context: str, max_chars: int = 500) -> str:
    """
    Compresses multi-layer memory context into a dense, high-signal summary
    (max ~150 tokens) containing key operator directives, rules, and project context.
    """
    if not memory_context or not memory_context.strip():
        return "Operator: Sir/AIvco | System: AIVCO Personal AI Architecture."

    lines = []
    for raw_line in memory_context.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip list markers
        clean_line = re.sub(r"^[-*•✓\s]+", "", line).strip()
        if any(key in clean_line.lower() for key in (
            "operator address", "operating rules", "brevity protocol",
            "active project", "tech stack", "role", "preferred"
        )):
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
        summary = summary[:max_chars - 3].rsplit(" ", 1)[0] + "..."
    return summary or "Operator: Sir/AIvco | System: AIVCO Personal AI Architecture."


def build_executive_envelope(prompt: str, target_tier: str, memory_context: str) -> str:
    """
    Wraps prompt with an Executive Context Envelope for Brain 4 Web AI models.
    For Tier 2 Cloud API models, keeps the user message clean while memory is passed via system context.
    """
    tier_norm = (target_tier or "").lower()
    compact_memory_summary = compress_memory_context(memory_context)

    # Brain 4 Web AI (Claude, ChatGPT, Gemini Web, Grok via Extension)
    if any(k in tier_norm for k in ("claude", "chatgpt", "gemini_web", "gemini-web", "grok", "brain4", "web")):
        envelope = (
            "[AIVCO EXECUTIVE BRIEF]\n"
            f"Operator Context: {compact_memory_summary}\n"
            "Execution Contract: Deliver production-grade, complete implementation. "
            "Do NOT use the Artifacts sandbox tool or create external file cards—output all code directly "
            "inline inside standard markdown fenced code blocks (e.g., ```html <!DOCTYPE html>... ```) "
            "so AIVCO's Live Preview Engine can render it immediately.\n"
            "[/AIVCO EXECUTIVE BRIEF]\n\n"
            "USER DIRECTIVE:\n"
            f"{prompt}"
        )
        return envelope

    # Tier 2 Fast Cloud API (Groq, Gemini) / Kernel / Local
    return prompt


# ── Kernel Cognitive Arbiter Class ───────────────────────────────────────────

class KernelCognitiveArbiter:
    # 1. Direct Kernel / System / Memory Patterns (< 5ms resolution)
    KERNEL_PATTERNS = [
        r"^(who am i|my profile|system status|kernel status|dataset stats|tasks left)\??$",
        r"^(clear|reset|clean) (screen|memory|chat|context)$",
        r"^(open|launch) (terminal|code|browser|dashboard)$",
        r"^(what is|calc) \d+[\+\-\*\/]\d+",
    ]

    # 2. Brain 4 Web AI Patterns (Complex full-stack HTML/JS apps, complete game engines, massive code)
    WEB_AI_PATTERNS = [
        r"(single-file|complete) (html|website|dashboard|game|canvas|three\.js|webpage)",
        r"(build|make|create|write) (a|an) (full|complete|interactive|tactical) (website|web page|dashboard|clone|app)",
        r"(build|make|create|write)\s+(?:a|an)\s+(?:(?:full|complete|interactive|tactical)\s+)+(?:website|web page|dashboard|clone|app)",
        r"<!doctype html>",
        r"tactical dashboard|starting xi|probability calculator",
    ]

    # 3. Tier 2 Fast Cloud API Patterns (Groq Llama 70B / Gemini Pro)
    API_FAST_PATTERNS = [
        r"(write|implement|explain|debug|refactor) (python|c\+\+|javascript|rust|sql|function|script|regex)",
        r"(analyze|summarize|translate|extract|parse) (this|the following|json|data)",
        r"(compare|contrast|differences between)",
        r"\b(why|how does|architecture of)\b",
    ]

    # Pre-compiled individual patterns
    _COMPILED_KERNEL = [re.compile(p, re.IGNORECASE) for p in KERNEL_PATTERNS]
    _COMPILED_WEB = [re.compile(p, re.IGNORECASE) for p in WEB_AI_PATTERNS]
    _COMPILED_API = [re.compile(p, re.IGNORECASE) for p in API_FAST_PATTERNS]

    # Pre-compiled combined patterns for sub-millisecond execution
    _FAST_KERNEL = re.compile("|".join(f"(?:{p})" for p in KERNEL_PATTERNS), re.IGNORECASE)
    _FAST_WEB = re.compile("|".join(f"(?:{p})" for p in WEB_AI_PATTERNS), re.IGNORECASE)
    _FAST_API = re.compile("|".join(f"(?:{p})" for p in API_FAST_PATTERNS), re.IGNORECASE)

    @classmethod
    def arbitrate(
        cls,
        prompt: str,
        target_override: Optional[str] = None,
        log_telemetry: bool = True,
    ) -> Tuple[str, str]:
        """
        Deterministically arbitrates prompt intent in <5ms (zero LLM calls).

        Returns:
            Tuple[str, str]: (target_tier, notification_message)
            Tiers: 'kernel', 'claude' (Brain 4 Web), 'groq' (Tier 2 Fast API), etc.
        """
        t0 = time.perf_counter()

        # If user explicitly locked a pill in UI, respect override
        if target_override and target_override.strip().lower() not in ["auto", "auto (arbiter)", ""]:
            tier = target_override
            rationale = f"[aivco-kernel] Manual override active: Routing to {target_override}."
        else:
            clean = prompt.strip().lower()

            # Check Kernel-native (< 5ms resolution)
            if cls._FAST_KERNEL.search(clean):
                tier = "kernel"
                rationale = ""
            # Check Web AI (Brain 4 Claude / ChatGPT via Extension)
            elif cls._FAST_WEB.search(clean):
                tier = "claude"
                rationale = "[aivco-kernel] Heavy frontend/application query detected. Escalating to Brain 4 (Claude Web Bridge)..."
            # Check Fast API (Tier 2 Groq / Gemini)
            elif cls._FAST_API.search(clean):
                tier = "groq"
                rationale = "[aivco-kernel] Analytical/coding query detected. Routing to Tier 2 (Groq Fast API)..."
            # Default fallback: Tier 2 (Groq) or Tier 1 (Local)
            else:
                tier = "groq"
                rationale = "[aivco-kernel] General query detected. Routing to Tier 2 (Fast Inference)..."

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if log_telemetry:
            log_arbiter_telemetry(prompt, tier, rationale, latency_ms)

        return tier, rationale

    @classmethod
    def flush_telemetry(cls, timeout: float = 2.0) -> None:
        """Flushes pending telemetry writes to disk."""
        flush_telemetry(timeout=timeout)
