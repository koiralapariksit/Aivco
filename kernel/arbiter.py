"""
AIVCO Kernel Cognitive Arbiter & Executive Context Broker
Deterministically classifies incoming user queries into Kernel (Zero-LLM),
Brain 4 Web AI (Claude/ChatGPT via Extension Bridge), or Tier 2 Fast API (Groq/Gemini).
Runs in <5ms with 0-LLM token consumption.
Broker provides executive context envelopes and asynchronous telemetry training logging.
"""

from datetime import datetime, timezone
import difflib
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

IDENTITY_PATTERN = re.compile(
    r"\b(about\s+me|who\s+am\s+i|know\s+(?:a?bout|abt)\s+me|my\s+profile|tell\s+(?:me\s+)?about\s+me|who\s+i\s+am)\b",
    re.IGNORECASE,
)
FRONTEND_PATTERN = re.compile(
    r"\b(web(?:site|asite)|e-?commer(?:ce|ice)|clone|dashboard|html|web\s*app|store|spotify|ui|interface|landing\s+page)\b",
    re.IGNORECASE,
)
CODING_PATTERN = re.compile(
    r"\b(py(?:thon|hton)|script|scirpt|tkinter|gui|algorithm|api|backend|endpoint|fastapi|function)\b",
    re.IGNORECASE,
)

KNOWN_PROVIDER_ALIASES: Dict[str, str] = {
    # Claude
    "claude": "claude", "cladue": "claude", "claud": "claude", "cloude": "claude",
    "calude": "claude", "claudeai": "claude", "claude_web": "claude",
    # ChatGPT
    "chatgpt": "chatgpt", "chat gpt": "chatgpt", "gpt": "chatgpt", "chagpt": "chatgpt",
    "chatgtp": "chatgpt", "openai": "chatgpt", "gpt4": "chatgpt", "chatgpt_web": "chatgpt",
    # Gemini
    "gemini web": "gemini_web", "gemini_web": "gemini_web", "gemini": "gemini_web",
    "gmini": "gemini_web", "gemni": "gemini_web", "gemeni": "gemini_web",
    # Grok
    "grok": "grok", "gork": "grok", "xai": "grok", "grok_web": "grok",
    # Fast API / Local
    "groq": "groq", "grq": "groq",
    "local": "local_llama", "local_llama": "local_llama", "llama": "local_llama",
    "kernel": "kernel", "aivco-kernel": "kernel",
}

CANONICAL_PROVIDERS: Dict[str, List[str]] = {
    "claude": ["claude"],
    "chatgpt": ["chatgpt", "gpt"],
    "gemini_web": ["gemini", "gemini web"],
    "grok": ["grok", "xai"],
}


DIRECTIVE_TRIGGER_CHECK = re.compile(
    r"\b(?:use|ask|tell|run|search|via|with|on|in|switch\s+to|@)\b|^[a-zA-Z0-9_\-]+:\s*",
    re.IGNORECASE,
)
TRIGGER_PATTERN = re.compile(
    r"^(?:(?:use|ask|tell|run|search|via|with|on|in)\s+)?@?([a-zA-Z0-9_\-]+(?:\s+web)?)(?:\s*(?:to\s+(?:search\s+)?|for\s+|:\s*|\s+-\s*|\s+–\s*)|\s+)(.*)$",
    re.IGNORECASE,
)


def match_provider_token(token: str) -> Optional[str]:
    """Matches a token to a provider canonical name using aliases and SequenceMatcher."""
    t = token.lower().strip()
    if t in KNOWN_PROVIDER_ALIASES:
        return KNOWN_PROVIDER_ALIASES[t]
    lt = len(t)
    if lt < 3 or lt > 12:
        return None
    for can, targets in CANONICAL_PROVIDERS.items():
        for target in targets:
            if abs(lt - len(target)) <= 2:
                if difflib.SequenceMatcher(None, t, target).ratio() >= 0.78:
                    return can
    return None


def extract_fuzzy_provider_directive(prompt: str) -> Tuple[Optional[str], str]:
    """
    Sub-millisecond fuzzy typo-tolerant provider directive extraction.
    Detects provider commands at start or inside prompt (e.g. 'use cladue to', 'use claudeto',
    'ask gpt to', 'use chagpt') and strips ONLY the routing prefix cleanly.
    """
    raw = prompt.strip()
    if not raw or not DIRECTIVE_TRIGGER_CHECK.search(raw):
        return None, raw

    # Check 1: Explicit switch commands ("switch to claude", "switch to grok")
    m_switch = re.match(r"^(?:switch\s+to\s+)([\w\s\-]+?)[.?!]*$", raw, re.IGNORECASE)
    if m_switch:
        prov = match_provider_token(m_switch.group(1))
        if prov:
            return prov, ""

    # Check 2: Prefix directives: (use|ask|tell|run|via|with|search)? @?(provider)(to|:|-) rest
    # Also handles merged 'to' (e.g. claudeto, cladueto, chagptto, gptto)
    m_pref = TRIGGER_PATTERN.match(raw)
    if m_pref:
        token = m_pref.group(1).strip()
        rest = m_pref.group(2).strip()

        # Check merged 'to' (e.g. claudeto -> base 'claude', rest)
        if token.lower().endswith("to") and token.lower() not in KNOWN_PROVIDER_ALIASES:
            base_token = token[:-2]
            prov = match_provider_token(base_token)
            if prov:
                return prov, rest

        prov = match_provider_token(token)
        if prov:
            return prov, rest

    # Check 3: Short standalone (e.g. 'use cladue', 'ask gpt', 'chagpt')
    m_standalone = re.match(r"^(?:(?:use|ask|tell)\s+)?@?([a-zA-Z0-9_\-]+(?:\s+web)?)[.?!]*$", raw, re.IGNORECASE)
    if m_standalone:
        token = m_standalone.group(1).strip()
        if token.lower().endswith("to") and token.lower() not in KNOWN_PROVIDER_ALIASES:
            base_token = token[:-2]
            prov = match_provider_token(base_token)
            if prov:
                return prov, ""
        prov = match_provider_token(token)
        if prov:
            return prov, ""

    # Check 4: Inline trailing directive: '... using cladue', '... with chagpt'
    m_inline = re.search(r"\b(?:using|with|via|in|on)\s+@?([a-zA-Z0-9_\-]+(?:\s+web)?)[.?!]*$", raw, re.IGNORECASE)
    if m_inline:
        prov = match_provider_token(m_inline.group(1).strip())
        if prov:
            clean = raw[:m_inline.start()].strip()
            return prov, clean

    return None, raw


def synthesize_elite_prompt(raw_intent: str, target_tier: str = "", memory_context: str = "") -> str:
    """
    Transforms casual or typo-laden operator requests into elite, production-grade
    engineering specifications before dispatching to AI providers.
    """
    clean_intent = raw_intent.strip()
    if not clean_intent:
        return clean_intent

    # 1. Identity / Personal queries
    if IDENTITY_PATTERN.search(clean_intent):
        return (
            "Analyze the provided Operator Context in the Executive Brief and provide a comprehensive, "
            "structured executive summary of who the operator (Parikshit Koirala / AIvco) is, including "
            "technical stack, active engineering projects, academic background, and design preferences."
        )

    # 2. Frontend / Web App / UI / Clone queries
    if FRONTEND_PATTERN.search(clean_intent):
        return (
            f"USER OBJECTIVE: {clean_intent}\n\n"
            "ELITE ENGINEERING SPECIFICATION:\n"
            "- Architect and output a complete, self-contained, single-file `<!DOCTYPE html>` application (HTML5 + modern CSS3 + vanilla JavaScript).\n"
            "- Design Aesthetic: Bespoke, human-crafted editorial or dark obsidian/glassmorphic UI with custom inline SVG artwork, micro-interactions, responsive CSS Grid/Flexbox layout, and zero broken external image links.\n"
            "- Functional Completeness: Implement working state management, interactive filters/search/controls, smooth drawers/modals, and realistic domain data.\n"
            "- Output Contract: Do NOT use the Artifacts sandbox tool or external file cards. Output the ENTIRE runnable code directly inline inside a single ```html ... ``` fenced code block."
        )

    # 3. Coding / Script / Backend queries
    if CODING_PATTERN.search(clean_intent):
        return (
            f"USER OBJECTIVE: {clean_intent}\n\n"
            "ELITE ENGINEERING SPECIFICATION:\n"
            "- Deliver clean, production-grade, well-structured, and immediately runnable code inside standard markdown fenced code blocks.\n"
            "- Handle edge cases, include clear inline documentation, and provide a concise architectural breakdown."
        )

    return clean_intent


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
    For Tier 2/3 Cloud API models, keeps the user message clean while memory is passed via system context.
    """
    tier_norm = (target_tier or "").lower()
    compact_memory_summary = compress_memory_context(memory_context)

    # Brain 4 Web AI (Claude, ChatGPT, Gemini Web, Grok via Extension)
    is_web_ai = any(k in tier_norm for k in ("claude", "chatgpt", "gemini_web", "gemini-web", "grok", "brain4")) or tier_norm == "web"
    if is_web_ai:
        synthesized_prompt = synthesize_elite_prompt(prompt, target_tier, memory_context)
        envelope = (
            "[AIVCO EXECUTIVE BRIEF]\n"
            f"Operator Context: {compact_memory_summary}\n"
            "Execution Contract: Deliver production-grade, complete implementation. "
            "Do NOT use the Artifacts sandbox tool or create external file cards—output all code directly "
            "inline inside standard markdown fenced code blocks (e.g., ```html <!DOCTYPE html>... ```) "
            "so AIVCO's Live Preview Engine can render it immediately.\n"
            "[/AIVCO EXECUTIVE BRIEF]\n\n"
            "USER DIRECTIVE:\n"
            f"{synthesized_prompt}"
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
        r"(build|make|create|write) (a|an)?\s*(full|complete|interactive|tactical)?\s*(website|web\s*page|dashboard|clone|app|store|e-?commerce)",
        r"(build|make|create|write)\s+(?:a|an)?\s+(?:(?:full|complete|interactive|tactical)\s+)+(?:website|web\s*page|dashboard|clone|app|store)",
        r"<!doctype html>",
        r"tactical dashboard|starting xi|probability calculator",
        r"spotify clone|netflix clone|amazon clone",
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
            # Check explicit fuzzy provider directive (e.g. 'use cladue to...', 'ask gpt to...', 'use chagpt')
            prov_directive, _ = extract_fuzzy_provider_directive(prompt)
            if prov_directive:
                tier = prov_directive
                rationale = "" if tier == "kernel" else f"[aivco-kernel] Operator directive detected: Routing to {tier}..."
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
