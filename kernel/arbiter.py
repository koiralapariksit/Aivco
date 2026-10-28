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
    r"\b(web(?:site|asite)|e-?commer(?:ce|ice)|clon(?:e|ing|ed)|claong|dashboard|html|web\s*app|store|spotify|netflix|nelflix|youtube|ui(?:\s+only)?|interface|landing\s+page|working\s+app|build\s+it|make\s+it)\b",
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
    # Gemini Cloud API (Tier 3 Fast Cloud API Key)
    "gemini": "gemini", "gemeni": "gemini", "gmini": "gemini", "gemni": "gemini",
    "gemini api": "gemini", "gemeni api": "gemini", "gemini key": "gemini", "gemeni key": "gemini",
    "gemini pro": "gemini", "gemini flash": "gemini", "gemeni pro": "gemini", "gemeni flash": "gemini",
    "gemini-flash": "gemini", "gemini-pro": "gemini",
    # Gemini Web (Browser Account Automation)
    "gemini web": "gemini_web", "gemini_web": "gemini_web", "gemini browser": "gemini_web",
    "gemeni web": "gemini_web", "gemeni browser": "gemini_web",
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
    "gemini": ["gemini", "gemeni", "gemini api", "gemeni api"],
    "gemini_web": ["gemini web", "gemeni web", "gemini browser"],
    "grok": ["grok", "xai"],
    "groq": ["groq"],
}


# ── Live Provider Health & Circuit Breaker ───────────────────────────────────

class ProviderHealthRecord:
    def __init__(self, name: str):
        self.name = name
        self.healthy: bool = True
        self.cooldown_until: float = 0.0
        self.consecutive_failures: int = 0
        self.avg_latency_ms: float = 0.0
        self.last_failure_reason: str = ""

    def is_available(self, extension_connected: bool = True) -> bool:
        if self.name in ("claude", "chatgpt", "gemini_web", "grok", "claude_web", "chatgpt_web", "grok_web"):
            if not extension_connected:
                return False
        if time.time() < self.cooldown_until:
            return False
        return True


class ProviderHealthRegistry:
    """
    Live Provider Health & Circuit Breaker.
    Prevents repeated dispatches to rate-limited, cloudflare-challenged, or disconnected providers.
    """
    DEFAULT_PROVIDERS = ["claude", "chatgpt", "gemini_web", "grok", "gemini", "groq"]

    def __init__(self):
        self._lock = threading.Lock()
        self._registry: Dict[str, ProviderHealthRecord] = {
            p: ProviderHealthRecord(p) for p in self.DEFAULT_PROVIDERS
        }
        self._has_active_cooldown: bool = False

    def _normalize(self, provider: str) -> str:
        prov = (provider or "").strip().lower()
        prov = prov.replace("-", "_")
        if prov in ("gemini_web", "claude_web", "chatgpt_web", "grok_web"):
            return prov
        if prov.startswith("web_account:") or prov.startswith("web-account:"):
            prov = prov.split(":", 1)[1]
        if prov in KNOWN_PROVIDER_ALIASES:
            return KNOWN_PROVIDER_ALIASES[prov]
        return prov

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

            if "rate_limit" in reason_lower or "rate limit" in reason_lower or "cloudflare" in reason_lower:
                rec.cooldown_until = now + 600.0  # 10 minutes
            elif "timeout" in reason_lower or "disconnected" in reason_lower or "disconnect" in reason_lower:
                rec.cooldown_until = now + 120.0  # 2 minutes
            else:
                if rec.consecutive_failures >= 2:
                    rec.cooldown_until = now + 60.0
                else:
                    rec.cooldown_until = now + 15.0

    def record_success(self, provider: str, latency_ms: float = 0.0) -> None:
        norm = self._normalize(provider)
        with self._lock:
            if norm not in self._registry:
                self._registry[norm] = ProviderHealthRecord(norm)
            rec = self._registry[norm]
            rec.consecutive_failures = 0
            rec.cooldown_until = 0.0
            rec.healthy = True
            rec.last_failure_reason = ""
            if latency_ms > 0:
                if rec.avg_latency_ms <= 0:
                    rec.avg_latency_ms = latency_ms
                else:
                    rec.avg_latency_ms = 0.7 * rec.avg_latency_ms + 0.3 * latency_ms

    def is_available(self, provider: str, extension_connected: bool = True) -> bool:
        if not self._has_active_cooldown and extension_connected:
            return True
        norm = self._normalize(provider)
        with self._lock:
            rec = self._registry.get(norm)
            if not rec:
                rec = ProviderHealthRecord(norm)
                self._registry[norm] = rec
            return rec.is_available(extension_connected)

    def get_best_available_provider(
        self,
        preferred_chain: list[str],
        extension_connected: bool = True,
    ) -> str:
        if not self._has_active_cooldown and extension_connected:
            return self._normalize(preferred_chain[0]) if preferred_chain else "groq"

        now = time.time()
        with self._lock:
            has_any_cd = False
            for p in preferred_chain:
                norm = self._normalize(p)
                rec = self._registry.get(norm)
                if not rec:
                    rec = ProviderHealthRecord(norm)
                    self._registry[norm] = rec

                if norm in ("claude", "chatgpt", "gemini_web", "grok", "claude_web", "chatgpt_web", "grok_web"):
                    if not extension_connected:
                        continue

                if now < rec.cooldown_until:
                    has_any_cd = True
                    continue

                return norm

            self._has_active_cooldown = has_any_cd

        return self._normalize(preferred_chain[-1]) if preferred_chain else "groq"

    def reset_all(self) -> None:
        with self._lock:
            for rec in self._registry.values():
                rec.healthy = True
                rec.cooldown_until = 0.0
                rec.consecutive_failures = 0
                rec.last_failure_reason = ""
            self._has_active_cooldown = False


provider_health = ProviderHealthRegistry()


DIRECTIVE_TRIGGER_CHECK = re.compile(
    r"\b(?:use|ask|tell|run|search|via|with|on|in|switch\s+to|@)\b|^[a-zA-Z0-9_\-]+:\s*",
    re.IGNORECASE,
)
TRIGGER_PATTERN = re.compile(
    r"^(?:(?:use|ask|tell|run|search|via|with|on|in)\s+)?@?([a-zA-Z0-9_\-]+(?:\s+(?:web|browser|api|key|pro|flash))?)(?:\s*(?:to\s+(?:search\s+)?|for\s+|:\s*|\s+-\s*|\s+–\s*)|\s+)(.*)$",
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
    m_standalone = re.match(r"^(?:(?:use|ask|tell)\s+)?@?([a-zA-Z0-9_\-]+(?:\s+(?:web|browser|api|key|pro|flash))?)[.?!]*$", raw, re.IGNORECASE)
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
    m_inline = re.search(r"\b(?:using|with|via|in|on)\s+@?([a-zA-Z0-9_\-]+(?:\s+(?:web|browser|api|key|pro|flash))?)[.?!]*$", raw, re.IGNORECASE)
    if m_inline:
        prov = match_provider_token(m_inline.group(1).strip())
        if prov:
            clean = raw[:m_inline.start()].strip()
            return prov, clean

    return None, raw


# ── Domain Feature Blueprints ───────────────────────────────────────────────

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
        lower_intent = clean_intent.lower()
        if re.search(r"\b(netflix|nelflix|youtube|video|stream(?:ing)?|cinema|movie)\b", lower_intent):
            blueprint = VIDEO_STREAMING_BLUEPRINT
        elif re.search(r"\b(spotify|music|audio|player|song|sound|playlist|track)\b", lower_intent):
            blueprint = MUSIC_AUDIO_BLUEPRINT
        elif re.search(r"\b(ecommerce|e-?commer(?:ce|ice)|store|shop|brand|sell|tshimhchet|cart|checkout|product)\b", lower_intent):
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
            "4. Even if the operator writes typos (e.g., 'claong website of nelflix', 'buikd', 'amek'), infer the exact target product and build the complete working clone on the first turn.\n\n"
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
            "Act with absolute CEO autonomy (ZERO hand-holding, no clarifying questions, no text roadmaps). "
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
        return {"ui_density": 0.0, "logic_depth": 0.0, "context_need": 0.0, "speed_priority": 1.0}

    # 1. Context Need (Zero-LLM kernel queries)
    context_need = 0.0
    if KernelCognitiveArbiter._FAST_KERNEL.search(clean) or IDENTITY_PATTERN.search(clean):
        context_need = 1.0
    elif any(k in clean for k in ("who am i", "my profile", "system status", "kernel status", "tasks left", "clear screen", "clear memory")):
        context_need = 1.0

    # 2. UI Density (Full-stack HTML/JS apps, complete game engines, websites, dashboards, clones)
    ui_density = 0.0
    if KernelCognitiveArbiter._FAST_WEB.search(clean):
        ui_density = 0.95
    elif FRONTEND_PATTERN.search(clean):
        if any(k in clean for k in ("clone", "claong", "single-file", "complete website", "dashboard", "ecommerce", "store", "player", "game", "canvas", "three.js", "netflix", "nelflix", "youtube", "spotify")):
            ui_density = 0.90
        elif any(k in clean for k in ("website", "webasite", "web page", "html", "app", "interface", "ui only", "working app")):
            ui_density = 0.80
        else:
            ui_density = 0.70

    # 3. Logic Depth (Algorithmic / Data Science / Python / SQL / Debugging)
    logic_depth = 0.0
    if KernelCognitiveArbiter._FAST_API.search(clean):
        logic_depth = 0.90
    elif CODING_PATTERN.search(clean):
        if ui_density < 0.70:
            logic_depth = 0.85
        else:
            logic_depth = 0.60
    elif any(k in clean for k in ("binary search", "algorithm", "regex", "sql", "fastapi", "refactor", "debug", "benchmark", "sort")):
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
    # 1. Direct Kernel / System / Memory Patterns (< 5ms resolution)
    KERNEL_PATTERNS = [
        r"^(who am i|my profile|system status|kernel status|dataset stats|tasks left)\??$",
        r"^(clear|reset|clean) (screen|memory|chat|context)$",
        r"^(open|launch) (terminal|code|browser|dashboard)$",
        r"^(what is|calc) \d+[\+\-\*\/]\d+",
    ]

    # 2. Brain 4 Web AI Patterns (Complex full-stack HTML/JS apps, complete game engines, massive code)
    WEB_AI_PATTERNS = [
        r"(single-file|complete|working)\s+(html|website|webasite|dashboard|game|canvas|three\.js|webpage|app)",
        r"(build|make|maek|create|write)\s+(?:me\s+)?(?:a|an)?\s*(?:(?:full|complete|interactive|tactical|working)\s+)*(?:clon(?:e|ing)|claong)?\s*(?:website|webasite|web\s*page|dashboard|clone|claong|app|store|e-?commer(?:ce|ice))",
        r"(build|make|maek|create|write)\s+(?:it|a\s+working\s+app|a\s+working\s+clone)",
        r"<!doctype html>",
        r"tactical dashboard|starting xi|probability calculator",
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

    # Pre-compiled individual patterns
    _COMPILED_KERNEL = [re.compile(p, re.IGNORECASE) for p in KERNEL_PATTERNS]
    _COMPILED_WEB = [re.compile(p, re.IGNORECASE) for p in WEB_AI_PATTERNS]
    _COMPILED_API = [re.compile(p, re.IGNORECASE) for p in API_FAST_PATTERNS]

    # Pre-compiled combined patterns for sub-millisecond execution
    _FAST_KERNEL = re.compile("|".join(f"(?:{p})" for p in KERNEL_PATTERNS), re.IGNORECASE)
    _FAST_WEB = re.compile("|".join(f"(?:{p})" for p in WEB_AI_PATTERNS), re.IGNORECASE)
    _FAST_API = re.compile("|".join(f"(?:{p})" for p in API_FAST_PATTERNS), re.IGNORECASE)

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

        Returns:
            Tuple[str, str]: (target_tier, notification_message)
            Tiers: 'kernel', 'claude' (Brain 4 Web), 'chatgpt' (Brain 4 Web), 'gemini', 'groq', etc.
        """
        t0 = time.perf_counter()

        # If user explicitly locked a pill in UI, respect override
        if target_override and target_override.strip().lower() not in ["auto", "auto (arbiter)", ""]:
            tier = target_override
            rationale = f"[aivco-kernel] Manual override active: Routing to {target_override}."
        else:
            clean = prompt.strip().lower()

            # Fast-path 1: Kernel-native (<0.003ms)
            if cls._FAST_KERNEL.search(clean):
                tier = "kernel"
                rationale = ""
            else:
                # Check explicit fuzzy provider directive only if directive triggers present
                prov_directive = None
                if DIRECTIVE_TRIGGER_CHECK.search(prompt):
                    prov_directive, _ = extract_fuzzy_provider_directive(prompt)

                if prov_directive:
                    if prov_directive == "kernel":
                        tier = "kernel"
                        rationale = ""
                    else:
                        if provider_health.is_available(prov_directive, extension_connected=extension_connected):
                            tier = prov_directive
                            rationale = f"[aivco-kernel] Operator directive detected: Routing to {tier}..."
                        else:
                            scores = compute_vector_scores(prompt)
                            if scores["ui_density"] >= 0.7:
                                chain = ["claude", "chatgpt", "gemini", "groq"]
                            else:
                                chain = ["groq", "gemini"]
                            tier = provider_health.get_best_available_provider(chain, extension_connected=extension_connected)
                            rationale = f"[aivco-kernel] Operator directive '{prov_directive}' on cooldown/unavailable. Auto-routed to {tier}."
                elif cls._FAST_WEB.search(clean):
                    chain = ["claude", "chatgpt", "gemini", "groq"]
                    tier = provider_health.get_best_available_provider(chain, extension_connected=extension_connected)
                    if tier == "claude":
                        rationale = "[aivco-kernel] Heavy frontend/application query detected. Escalating to Brain 4 (Claude Web Bridge)..."
                    elif tier == "chatgpt":
                        rationale = "[aivco-kernel] Heavy frontend query detected. Claude on cooldown/unavailable; routing to Brain 4 (ChatGPT Web Bridge)..."
                    elif tier == "gemini":
                        rationale = "[aivco-kernel] Heavy frontend query detected. Web bridge on cooldown/unavailable; routing to Tier 3 (Gemini Pro API)..."
                    else:
                        rationale = "[aivco-kernel] Heavy frontend query detected. Cascading to Tier 2 (Groq Fast API)..."
                elif cls._FAST_API.search(clean):
                    chain = ["groq", "gemini", "claude"]
                    tier = provider_health.get_best_available_provider(chain, extension_connected=extension_connected)
                    rationale = f"[aivco-kernel] Analytical/coding query detected. Routing to Tier 2 ({tier.upper()})..."
                else:
                    # Multi-Vector Capability Matrix Scoring
                    scores = compute_vector_scores(prompt)
                    if scores["context_need"] >= 0.8:
                        tier = "kernel"
                        rationale = ""
                    elif scores["ui_density"] >= 0.7:
                        chain = ["claude", "chatgpt", "gemini", "groq"]
                        tier = provider_health.get_best_available_provider(chain, extension_connected=extension_connected)
                        if tier == "claude":
                            rationale = "[aivco-kernel] Heavy frontend/application query detected. Escalating to Brain 4 (Claude Web Bridge)..."
                        elif tier == "chatgpt":
                            rationale = "[aivco-kernel] Heavy frontend query detected. Claude on cooldown/unavailable; routing to Brain 4 (ChatGPT Web Bridge)..."
                        elif tier == "gemini":
                            rationale = "[aivco-kernel] Heavy frontend query detected. Web bridge on cooldown/unavailable; routing to Tier 3 (Gemini Pro API)..."
                        else:
                            rationale = "[aivco-kernel] Heavy frontend query detected. Cascading to Tier 2 (Groq Fast API)..."
                    elif scores["logic_depth"] >= 0.7:
                        chain = ["groq", "gemini", "claude"]
                        tier = provider_health.get_best_available_provider(chain, extension_connected=extension_connected)
                        rationale = f"[aivco-kernel] Analytical/coding query detected. Routing to Tier 2 ({tier.upper()})..."
                    else:
                        chain = ["groq", "gemini"]
                        tier = provider_health.get_best_available_provider(chain, extension_connected=extension_connected)
                        rationale = f"[aivco-kernel] General query detected. Routing to Tier 2 ({tier.upper()})..."

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if log_telemetry:
            log_arbiter_telemetry(prompt, tier, rationale, latency_ms)

        return tier, rationale

    @classmethod
    def flush_telemetry(cls, timeout: float = 2.0) -> None:
        """Flushes pending telemetry writes to disk."""
        flush_telemetry(timeout=timeout)


# ── Pre-Delivery Code Quality Gate & TypeScript-in-HTML Sanitizer ────────────

def sanitize_script_content(js_code: str) -> str:
    """
    Sanitizes inline JavaScript by stripping accidental TypeScript syntax:
    - type / interface declarations
    - type assertions ('as HTMLElement', 'as any', etc.)
    - non-null assertions ('!.addEventListener')
    - parameter and variable type annotations
    """
    if not js_code or not isinstance(js_code, str):
        return js_code or ""

    # 1. Strip interface declarations: interface Foo { ... }
    js_code = re.sub(
        r"interface\s+[A-Za-z0-9_$]+(?:\s+extends\s+[^{]+)?\s*\{[\s\S]*?\}",
        "",
        js_code,
    )

    # 2. Strip type declarations: type Foo = ...;
    js_code = re.sub(
        r"type\s+[A-Za-z0-9_$]+\s*=\s*\{[\s\S]*?\};?",
        "",
        js_code,
    )
    js_code = re.sub(
        r"type\s+[A-Za-z0-9_$]+\s*=\s*[^;\n]+;",
        "",
        js_code,
    )

    # 3. Strip type assertions: " as HTMLElement", " as HTMLInputElement", " as HTMLButtonElement", " as any", etc.
    js_code = re.sub(
        r"\s+as\s+[A-Za-z0-9_$]+(?:<[^>]+>)?(?:\[\])?",
        "",
        js_code,
    )

    # 4. Strip non-null assertions: "element!.addEventListener", "container!.innerHTML", "el!;"
    js_code = re.sub(
        r"([a-zA-Z0-9_$\]\)])!([.\[(])",
        r"\1\2",
        js_code,
    )
    js_code = re.sub(
        r"([a-zA-Z0-9_$\]\)])!\s*;",
        r"\1;",
        js_code,
    )

    # 5. Strip return type annotations on functions: function foo(...): Type {
    js_code = re.sub(
        r"\)\s*:\s*[A-Za-z0-9_$<>[\]|&?\s]+\s*\{",
        ") {",
        js_code,
    )

    # 6. Strip variable type annotations: const x: number = ..., let y: Item[] = ...
    js_code = re.sub(
        r"\b(const|let|var)\s+([a-zA-Z0-9_$]+)\s*:\s*[A-Za-z0-9_$<>[\]|&?\s]+(?=\s*=)",
        r"\1 \2",
        js_code,
    )

    # 7. Function parameters type annotations: function foo(param: Type, ...)
    def _clean_params(m):
        fn_name = m.group(1)
        raw_params = m.group(2)
        cleaned = []
        for p in raw_params.split(","):
            p_strip = p.strip()
            if not p_strip:
                continue
            if ":" in p_strip:
                p_name, rest = p_strip.split(":", 1)
                p_name = p_name.strip()
                if "=" in rest:
                    val = rest.split("=", 1)[1].strip()
                    cleaned.append(f"{p_name} = {val}")
                else:
                    cleaned.append(p_name)
            else:
                cleaned.append(p_strip)
        return f"function {fn_name}({', '.join(cleaned)})"

    js_code = re.sub(
        r"function\s*([a-zA-Z0-9_$]*)\s*\(([^)]*)\)",
        _clean_params,
        js_code,
    )

    # 8. Arrow function return types and parameters: (a: number, b: number): number =>
    js_code = re.sub(
        r"\(([^)]*)\)\s*:\s*[A-Za-z0-9_$<>[\]|&?\s]+\s*=>",
        r"(\1) =>",
        js_code,
    )

    def _clean_arrow_params(m):
        raw_params = m.group(1)
        cleaned = []
        for p in raw_params.split(","):
            p_strip = p.strip()
            if not p_strip:
                continue
            if ":" in p_strip:
                p_name = p_strip.split(":", 1)[0].strip()
                cleaned.append(p_name)
            else:
                cleaned.append(p_strip)
        return f"({', '.join(cleaned)}) =>"

    js_code = re.sub(
        r"\(([^)]+:[^)]+)\)\s*=>",
        _clean_arrow_params,
        js_code,
    )

    return js_code.strip()


def sanitize_and_validate_code_output(markdown_text: str) -> str:
    """
    Quality gate that cleans TypeScript syntax out of plain <script> blocks
    and ensures balanced HTML structure and markdown code fences.
    """
    if not markdown_text or not isinstance(markdown_text, str):
        return markdown_text or ""

    def _clean_script_tag(match):
        open_tag = match.group(1)
        body = match.group(2)
        close_tag = match.group(3)
        return f"{open_tag}\n{sanitize_script_content(body)}\n{close_tag}"

    sanitized = re.sub(
        r"(<script(?:\s+[^>]*)?>)([\s\S]*?)(</script>)",
        _clean_script_tag,
        markdown_text,
        flags=re.IGNORECASE,
    )

    # Ensure unclosed ```html block or missing </html> is closed
    if "```html" in sanitized.lower():
        lower_san = sanitized.lower()
        if "<html" in lower_san and "</html>" not in lower_san:
            last_fence = sanitized.rfind("```")
            if last_fence > sanitized.lower().find("```html"):
                sanitized = sanitized[:last_fence] + "\n</html>\n" + sanitized[last_fence:]
            else:
                sanitized += "\n</html>\n```"
        else:
            fence_count = sanitized.count("```")
            if fence_count % 2 != 0:
                sanitized += "\n```"

    return sanitized
