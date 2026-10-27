"""
AIVCO Cognitive Kernel — Zero-LLM First Intelligence Layer

Architecture:
  user_prompt
      |
      v
  CognitiveKernel.evaluate_query()   <-- Subsystem 1 (Stage 0/1)
      |
      +-- confidence >= 0.85  --> [aivco-kernel] answer locally (<5ms, 0 LLM tokens)
      |
      +-- confidence < 0.85   --> Subsystem 3: Autonomous Model Arbiter
              |
              +--> scrub_sensitive_context()   (Privacy redaction)
              |
              +--> smart tier selection:
                     "private"  -> Tier 1 local_llama (strict lock)
                     "deep"     -> Tier 3 gemini
                     "general"  -> Tier 2 groq (default fast path)

  After every LLM turn:
      learn_from_turn()  <-- Subsystem 2 (Continuous Learning)

Subsystem 4:
  FineTuneDatasetLogger  -> memory/dataset_logger.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from kernel.arbiter import (
    KernelCognitiveArbiter,
    build_executive_envelope,
    compress_memory_context,
)

if TYPE_CHECKING:
    from memory.multi_layer_memory import MultiLayerMemory


# ---------------------------------------------------------------------------
# Privacy / Sensitivity helpers
# ---------------------------------------------------------------------------

_SENSITIVE_KEYS = frozenset({
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "credential", "private_key", "private key", "auth", "bearer",
})

_PRIVATE_PATTERNS = [
    re.compile(r"\bpassword\b", re.IGNORECASE),
    re.compile(r"\bsecret\b", re.IGNORECASE),
    re.compile(r"\bapi.?key\b", re.IGNORECASE),
    re.compile(r"\btoken\b", re.IGNORECASE),
    re.compile(r"\b(gsk_|AIzaSy)[A-Za-z0-9\-_]{8,}\b"),  # raw Groq/Gemini keys
    re.compile(r"\[private\]", re.IGNORECASE),
    re.compile(r"\bpasswd\b", re.IGNORECASE),
    re.compile(r"\bprivate.key\b", re.IGNORECASE),
]

_SENSITIVE_PROMPT_TRIGGERS = frozenset({
    "password", "my key", "api key", "secret", "token", "private", "credentials",
    "health", "diary", "finance", "banking", "wallet", "salary",
})

_DEEP_REASONING_TRIGGERS = frozenset({
    "architect", "architecture", "mathematical proof", "compare trade-offs",
    "tradeoff", "trade-off", "system design", "step-by-step analysis",
    "analyze", "deep dive", "prove that", "formal proof", "complexity analysis",
    "big-o", "data science", "ml pipeline", "research", "whitepaper",
})


def scrub_sensitive_context(context_str: str) -> str:
    """
    Remove or redact any lines/entries from context_str that contain
    sensitive/private credentials before sending to cloud providers.
    """
    if not context_str:
        return context_str
    lines = context_str.splitlines()
    clean: list[str] = []
    for line in lines:
        if any(pat.search(line) for pat in _PRIVATE_PATTERNS):
            # Replace the whole line with a redaction notice
            clean.append("[REDACTED — private/sensitive content]")
        else:
            clean.append(line)
    return "\n".join(clean)


# ---------------------------------------------------------------------------
# Subsystem 1 — Zero-LLM Cognitive Resolver
# ---------------------------------------------------------------------------

@dataclass
class KernelEvaluation:
    can_answer_locally: bool
    confidence: float                # 0.0 to 1.0
    local_response: Optional[str]
    recommended_tier: str            # "aivco-kernel" | "tier1" | "tier2" | "tier3"
    reason: str
    task_type: str = "general"       # used by arbiter downstream


# --- Natural language patterns for memory-backed answers ---

_WHO_AM_I = re.compile(
    r"\b(who\s+am\s+i|my\s+(name|identity)|what\s+is\s+my\s+name|"
    r"tell\s+me\s+about\s+myself|introduce\s+yourself\s+to\s+me|"
    r"about\s+me|my\s+profile)\b",
    re.IGNORECASE,
)

_TECH_STACK = re.compile(
    r"\b(my\s+tech\s+(stack|skills?|tools?)|what\s+(do\s+i|can\s+i)\s+(use|know|code|program)|"
    r"my\s+(languages?|frameworks?|coding|programming))\b",
    re.IGNORECASE,
)

_ROLE_QUERY = re.compile(
    r"\b(my\s+role|what\s+is\s+my\s+role|what\s+do\s+i\s+do|"
    r"what\s+am\s+i\s+studying|my\s+job|my\s+work|my\s+position)\b",
    re.IGNORECASE,
)

_PROJECT_QUERY = re.compile(
    r"\b(what\s+am\s+i\s+working\s+on|my\s+(current\s+)?(project|task|work)|"
    r"active\s+project|project\s+status|what\s+are\s+my\s+tasks|"
    r"pending\s+tasks|(?:what\s+)?(tasks?|milestones?)\s+(?:are\s+)?(?:left|remain|pending)|"
    r"what\s+have\s+i\s+(done|completed|finished))\b",
    re.IGNORECASE,
)

_EPISODE_QUERY = re.compile(
    r"\b(recent\s+(episodes?|history|activity|events?|progress)|"
    r"what\s+(did\s+we|have\s+we)\s+do\s+(recently|lately|last|before)|"
    r"show\s+(me\s+)?(recent|my|last)\s+(episodes?|history|events?)|"
    r"what\s+happened\s+recently)\b",
    re.IGNORECASE,
)

_LOCATION_QUERY = re.compile(
    r"\b(where\s+(am\s+i|do\s+i\s+live|are\s+you\s+from|am\s+i\s+from)|"
    r"my\s+location|my\s+(city|country|place))\b",
    re.IGNORECASE,
)

_EDUCATION_QUERY = re.compile(
    r"\b(my\s+(education|degree|university|college|school|study|studies|course)|"
    r"where\s+(do\s+i|am\s+i)\s+study|what\s+(am\s+i|do\s+i)\s+study)\b",
    re.IGNORECASE,
)


def _build_identity_response(memory: "MultiLayerMemory", full_profile: bool = False) -> Optional[str]:
    """Format an adaptive identity summary or full profile."""
    cp = memory.core_profile
    if not cp:
        return None

    preferred_address = memory.get_core_fact("preferred_address") or "Sir"

    if not full_profile:
        # Sharp 2-sentence tactical response respecting preferred_address
        return (
            f"You are AIvco ({preferred_address}) — my operator and system architect. "
            "First-year Data Science student at Kathmandu University and AI engineer. "
            "(Type 'full profile' anytime to inspect all 18 memory attributes.)"
        )

    # Full profile formatted list
    lines: list[str] = [
        f"### AIVCO Core Operator Profile (18 Attributes) — [{preferred_address}]:\n"
    ]
    for key, fact in sorted(cp.items()):
        val = fact.value
        title = key.replace("_", " ").title()
        lines.append(f"- **{title}**: {val}")
    return "\n".join(lines)


def _intercept_operator_directives(prompt: str, memory: "MultiLayerMemory") -> Optional[str]:
    """
    Detects natural language operator directives (name address, brevity, operating rules).
    Mutates MultiLayerMemory immediately and returns a confirmation string.
    Returns None if not a directive.
    """
    lower = prompt.lower()
    directives_found: list[str] = []

    # 1. Address / Name Override
    has_name_directive = any(phrase in lower for phrase in (
        "dont call me", "don't call me", "do not call me", "call me", 
        "address me as", "tell me sir", "allowed to tell me"
    ))

    if has_name_directive:
        target_addr = "sir"
        if "sir" in lower:
            target_addr = "sir"
        elif "aivco" in lower:
            target_addr = "AIvco"
        else:
            m = re.search(r"\b(?:call me|address me as|tell me)\s+([a-zA-Z]+)", prompt, re.IGNORECASE)
            if m:
                target_addr = m.group(1).lower()

        memory.set_core_fact("preferred_address", target_addr)

        avoid_name = "Parikshit"
        m_avoid = re.search(r"(?:dont|don't|do not)\s+call\s+me\s+([a-zA-Z]+)", prompt, re.IGNORECASE)
        if m_avoid:
            avoid_name = m_avoid.group(1).capitalize()

        disp_addr = "Sir" if target_addr.lower() == "sir" else target_addr
        rule = f"Never call the operator {avoid_name}; address him strictly as {disp_addr} or AIvco."
        current_rules = memory.get_core_fact("rules") or ""
        if rule not in current_rules:
            new_rules = f"{current_rules}; {rule}".strip("; ")
            memory.set_core_fact("rules", new_rules)

        directives_found.append(f"Preferred address set to '{disp_addr}'")

    # 2. Brevity / Casual Length Override
    has_brevity_directive = any(phrase in lower for phrase in (
        "avoid long answer", "avoid long answers", "keep replies short", 
        "keep reply short", "be concise", "avoid long explanation",
        "short answer in casual"
    ))

    if has_brevity_directive:
        memory.set_core_fact("concise_mode", "True")
        brevity_rule = "Keep casual chat and identity responses strictly concise (1-2 sentences) unless deep code or full profile is requested."
        current_rules = memory.get_core_fact("rules") or ""
        if brevity_rule not in current_rules:
            new_rules = f"{current_rules}; {brevity_rule}".strip("; ")
            memory.set_core_fact("rules", new_rules)
        directives_found.append("Brevity protocol enabled (concise responses for casual chat)")

    # 3. Explicit "teach <trigger> => <fact>"
    if "=>" in prompt and lower.strip().startswith("teach "):
        parts = prompt[6:].split("=>", 1)
        trig = parts[0].strip()
        fact = parts[1].strip()
        if trig and fact:
            if not hasattr(memory, "learned_facts"):
                memory.learned_facts = []
            memory.learned_facts.append({
                "trigger": trig,
                "trigger_keywords": [k for k in re.findall(r"\b\w+\b", trig.lower()) if len(k) >= 3],
                "fact": fact,
                "fact_type": "taught_rule",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hit_count": 0,
            })
            directives_found.append(f"Permanent rule: '{trig}' -> '{fact}'")

    if directives_found:
        memory.save()
        pref_val = memory.get_core_fact("preferred_address") or "Sir"
        pref = "Sir" if str(pref_val).lower() == "sir" else str(pref_val)
        summary = " & ".join(directives_found)
        return f"[aivco-kernel] Directive locked into Core Memory, {pref}: {summary}."

    return None


def detect_explicit_model_and_intent(prompt: str) -> dict:
    """
    Typo-tolerant natural language model and intent detector (Brain 1, 2, 3, 4).
    Detects explicit provider overrides, web search intent, explicit browser command,
    and stateful model lock commands.
    """
    raw = prompt.strip()
    lower = raw.lower()

    # 1. Physical desktop browser window ONLY if explicit command
    m_browser_cmd = re.match(r"^open\s+browser(?:\s+(.+))?$", raw, re.IGNORECASE)
    if m_browser_cmd:
        target = (m_browser_cmd.group(1) or "google").strip()
        return {
            "target_provider": None,
            "browser_target": target,
            "is_desktop_browser_request": True,
            "run_web_search": False,
            "clean_prompt": target,
            "is_model_switch_only": False,
            "switch_provider": None,
        }

    # 2. Model switch only ("switch to gemini", "switch to groq", etc.)
    m_switch = re.fullmatch(r"switch\s+to\s+(claude|chatgpt|grok|gemini_web|gemini|gemeni|groq|local|local_llama|kernel)[.?!]*", lower)
    if m_switch:
        prov = m_switch.group(1)
        p = "gemini" if ("gemen" in prov or "gemin" in prov and "web" not in prov) else ("local_llama" if "local" in prov else prov)
        return {
            "target_provider": p,
            "browser_target": None,
            "is_desktop_browser_request": False,
            "run_web_search": False,
            "clean_prompt": "",
            "is_model_switch_only": True,
            "switch_provider": p,
        }

    target_provider: Optional[str] = None
    run_web_search: bool = False
    clean_prompt: str = raw

    # 3. Model prefix extraction (e.g. 'search claude - ...', 'use claude to search...', 'claude: ...')
    m_model_prefix = re.match(
        r"^(?:search|seach|ask|use)\s+(claude|chatgpt|grok|gemini\s+web|gemini_web|gemini|gemeni|groq)\s*(?:to\s+(?:search|seach)\s*)?[-:]?\s*(.*)$",
        raw,
        re.IGNORECASE,
    )
    if m_model_prefix:
        model = m_model_prefix.group(1).lower().replace(" ", "_")
        target_provider = "gemini" if ("gemen" in model or model == "gemini") else model
        clean_prompt = m_model_prefix.group(2).strip()
        if re.search(r"^(?:search|seach)\s+", lower) or "to search" in lower or "to seach" in lower:
            run_web_search = True
    else:
        m_colon = re.match(r"^(claude|chatgpt|grok|gemini_web|gemini|gemeni|groq):\s*(.*)$", raw, re.IGNORECASE)
        if m_colon:
            model = m_colon.group(1).lower()
            target_provider = "gemini" if ("gemen" in model or model == "gemini") else model
            clean_prompt = m_colon.group(2).strip()

    # 4. Standalone model triggers if not matched by prefix
    if not target_provider:
        if re.search(r"\b(?:use\s+gemini\s+web|ask\s+gemini\s+web|@gemini_web)\b", lower):
            target_provider = "gemini_web"
            clean_prompt = re.sub(r"^(?:use\s+gemini\s+web|ask\s+gemini\s+web|@gemini_web)\s*(?:to|for)?\s*", "", clean_prompt, flags=re.IGNORECASE).strip()
        elif re.search(r"\b(?:use\s+(?:gemini|gemeni)|ask\s+(?:gemini|gemeni)|@(?:gemini|gemeni))\b", lower):
            target_provider = "gemini"
            clean_prompt = re.sub(r"^(?:use\s+(?:gemini|gemeni)|ask\s+(?:gemini|gemeni)|@(?:gemini|gemeni))\s*(?:to|for)?\s*", "", clean_prompt, flags=re.IGNORECASE).strip()
        elif re.search(r"\b(?:use\s+groq|ask\s+groq|@groq)\b", lower):
            target_provider = "groq"
            clean_prompt = re.sub(r"^(?:use\s+groq|ask\s+groq|@groq)\s*(?:to|for)?\s*", "", clean_prompt, flags=re.IGNORECASE).strip()
        elif re.search(r"\b(?:use\s+claude|ask\s+claude|@claude)\b", lower):
            target_provider = "claude"
            clean_prompt = re.sub(r"^(?:use\s+claude|ask\s+claude|@claude)\s*(?:to|for)?\s*", "", clean_prompt, flags=re.IGNORECASE).strip()
        elif re.search(r"\b(?:use\s+chatgpt|ask\s+chatgpt|@chatgpt)\b", lower):
            target_provider = "chatgpt"
            clean_prompt = re.sub(r"^(?:use\s+chatgpt|ask\s+chatgpt|@chatgpt)\s*(?:to|for)?\s*", "", clean_prompt, flags=re.IGNORECASE).strip()
        elif re.search(r"\b(?:use\s+grok|ask\s+grok|@grok)\b", lower):
            target_provider = "grok"
            clean_prompt = re.sub(r"^(?:use\s+grok|ask\s+grok|@grok)\s*(?:to|for)?\s*", "", clean_prompt, flags=re.IGNORECASE).strip()
        elif re.search(r"\b(?:use\s+local|@local|^private:|^offline:)\b", lower):
            target_provider = "local_llama"
            clean_prompt = re.sub(r"^(?:use\s+local|@local|private:|offline:)\s*", "", clean_prompt, flags=re.IGNORECASE).strip()
        elif re.search(r"\b(?:use\s+kernel|@kernel)\b", lower):
            target_provider = "aivco-kernel"
            clean_prompt = re.sub(r"^(?:use\s+kernel|@kernel)\s*", "", clean_prompt, flags=re.IGNORECASE).strip()

    # 5. Search triggers
    search_m = re.search(r"\b(?:@search|search\s+web|search\s+the\s+web|seach\s+web|seach\s+the\s+web|search\s+online|seach\s+online)\b", lower)
    if search_m:
        run_web_search = True
        clean_prompt = re.sub(r"^(?:@search|search\s+(?:the\s+)?web|seach\s+(?:the\s+)?web|search\s+online|seach\s+online)\s*(?:for|about|:|-)?\s*", "", clean_prompt, flags=re.IGNORECASE).strip()

    # 6. Automatic Factual Search Trigger
    is_identity = any(phrase in lower for phrase in (
        "who am i", "who are you", "who are u", "what is your name", "what is my name",
        "project status", "episode list", "dataset stats", "kernel status"
    ))
    if not is_identity and not run_web_search:
        if re.search(r"\b(search|seach|latest|current|most|who\s+(?:won|has|is)|record\s+for|records\s+for|award|awards|ballon\s+d'?or|champions\s+league|world\s+cup|presidential|election|olympic|olympics|2024|2025|2026)\b", clean_prompt, re.IGNORECASE):
            run_web_search = True

    return {
        "target_provider": target_provider,
        "browser_target": None,
        "is_desktop_browser_request": False,
        "run_web_search": run_web_search,
        "clean_prompt": clean_prompt.strip("-: \t\n") or raw,
        "is_model_switch_only": False,
        "switch_provider": None,
    }


def _build_project_response(memory: "MultiLayerMemory") -> Optional[str]:
    """Format active project status from Layer 3."""
    active_id = memory.get_active_project()
    if not active_id:
        return "No active project is currently set."
    proj = memory.projects.get(active_id)
    if not proj:
        return f"Active project '{active_id}' not found in memory."

    stack_str = ", ".join(proj.tech_stack) if proj.tech_stack else "N/A"
    lines = [
        f"Active Project: {proj.title} ({proj.project_id}) — {proj.status}",
        f"Tech Stack: {stack_str}",
        f"Completed ({len(proj.completed_milestones)}):",
    ]
    for m in proj.completed_milestones[-3:]:  # show last 3
        lines.append(f"  ✓ {m}")
    lines.append(f"Pending ({len(proj.pending_tasks)}):")
    for t in proj.pending_tasks:
        lines.append(f"  • {t}")
    return "\n".join(lines)


def _build_episode_response(memory: "MultiLayerMemory") -> Optional[str]:
    """Format recent episodic memory entries from Layer 2."""
    eps = memory.get_recent_episodes(limit=5)
    if not eps:
        return "No recent episodes recorded yet."
    lines = ["Recent activity:"]
    for ep in eps:
        date = ep.timestamp[:10] if ep.timestamp else "unknown"
        proj_tag = f" [{ep.project_id}]" if ep.project_id else ""
        lines.append(f"  [{date}]{proj_tag} {ep.event}")
    return "\n".join(lines)


def _get_core_field(memory: "MultiLayerMemory", *keys: str) -> Optional[str]:
    """Return value from the first matching core profile key."""
    for k in keys:
        fact = memory.core_profile.get(k)
        if fact:
            return fact.value
    return None


def evaluate_query(user_prompt: str, memory: "MultiLayerMemory") -> dict:
    """
    Stage 0 / Stage 1: Evaluate whether AIVCO can answer locally.

    Returns a dict with keys:
      can_answer_locally: bool
      confidence:         float  (0.0 to 1.0)
      local_response:     str | None
      recommended_tier:   str
      reason:             str
      task_type:          str
    """
    prompt = user_prompt.strip()
    lower = prompt.lower()

    # ---- Stage 0: Intercept Direct Operator Directives (Name override, brevity, rules) ----
    directive_response = _intercept_operator_directives(prompt, memory)
    if directive_response:
        return KernelEvaluation(
            can_answer_locally=True,
            confidence=0.99,
            local_response=directive_response,
            recommended_tier="aivco-kernel",
            reason="Operator directive intercepted and locked into Core Memory.",
        ).__dict__

    # ---- Stage 0: Explicit offline/private prefix ----
    if lower.startswith(("private:", "offline:", "memory:")):
        return KernelEvaluation(
            can_answer_locally=False,
            confidence=0.0,
            local_response=None,
            recommended_tier="tier1",
            reason="Private/offline prefix detected — routing to local Tier 1 only.",
            task_type="private",
        ).__dict__

    # ---- Stage 0: Explicit deep/research prefix ----
    if lower.startswith(("deep:", "research:")):
        return KernelEvaluation(
            can_answer_locally=False,
            confidence=0.0,
            local_response=None,
            recommended_tier="tier3",
            reason="Deep/research prefix detected — routing to Tier 3 Gemini.",
            task_type="deep",
        ).__dict__

    # ---- Stage 1: Full Profile Inspection ----
    if any(q in lower for q in ("full profile", "all details", "show full profile", "inspect profile")):
        resp = _build_identity_response(memory, full_profile=True)
        if resp:
            return KernelEvaluation(
                can_answer_locally=True,
                confidence=0.98,
                local_response=resp,
                recommended_tier="aivco-kernel",
                reason="Full profile requested from Core Profile (Layer 1).",
            ).__dict__

    # ---- Stage 1: Identity / Who am I ----
    if _WHO_AM_I.search(prompt):
        has_full = any(w in lower for w in ("full", "detail", "all"))
        resp = _build_identity_response(memory, full_profile=has_full)
        if resp:
            return KernelEvaluation(
                can_answer_locally=True,
                confidence=0.95,
                local_response=resp,
                recommended_tier="aivco-kernel",
                reason="Identity query resolved from Core Profile (Layer 1).",
            ).__dict__

    # ---- Stage 1: Tech stack ----
    if _TECH_STACK.search(prompt):
        val = _get_core_field(memory, "technical_stack", "languages", "stack", "tools")
        if val:
            return KernelEvaluation(
                can_answer_locally=True,
                confidence=0.93,
                local_response=f"Your tech stack: {val}",
                recommended_tier="aivco-kernel",
                reason="Tech stack query resolved from Core Profile.",
            ).__dict__

    # ---- Stage 1: Role / what do I do ----
    if _ROLE_QUERY.search(prompt):
        val = _get_core_field(memory, "role", "career_direction", "occupation")
        if val:
            return KernelEvaluation(
                can_answer_locally=True,
                confidence=0.93,
                local_response=f"Your role: {val}",
                recommended_tier="aivco-kernel",
                reason="Role query resolved from Core Profile.",
            ).__dict__

    # ---- Stage 1: Location ----
    if _LOCATION_QUERY.search(prompt):
        val = _get_core_field(memory, "location", "city", "country")
        if val:
            return KernelEvaluation(
                can_answer_locally=True,
                confidence=0.93,
                local_response=f"Your location: {val}",
                recommended_tier="aivco-kernel",
                reason="Location query resolved from Core Profile.",
            ).__dict__

    # ---- Stage 1: Education ----
    if _EDUCATION_QUERY.search(prompt):
        degree = _get_core_field(memory, "education_degree", "education", "degree")
        school = _get_core_field(memory, "education_institution", "university", "school")
        if degree or school:
            parts = []
            if degree:
                parts.append(f"Degree: {degree}")
            if school:
                parts.append(f"Institution: {school}")
            return KernelEvaluation(
                can_answer_locally=True,
                confidence=0.92,
                local_response="\n".join(parts),
                recommended_tier="aivco-kernel",
                reason="Education query resolved from Core Profile.",
            ).__dict__

    # ---- Stage 1: Project / task query ----
    if _PROJECT_QUERY.search(prompt):
        resp = _build_project_response(memory)
        if resp:
            return KernelEvaluation(
                can_answer_locally=True,
                confidence=0.92,
                local_response=resp,
                recommended_tier="aivco-kernel",
                reason="Project status query resolved from Layer 3 (Projects).",
            ).__dict__

    # ---- Stage 1: Episode / recent history ----
    if _EPISODE_QUERY.search(prompt):
        resp = _build_episode_response(memory)
        if resp:
            return KernelEvaluation(
                can_answer_locally=True,
                confidence=0.90,
                local_response=resp,
                recommended_tier="aivco-kernel",
                reason="Recent history query resolved from Layer 2 (Episodes).",
            ).__dict__

    # ---- Stage 1: Learned facts cache ----
    learned = getattr(memory, "learned_facts", None)
    if learned:
        prompt_words = set(re.findall(r"\b\w+\b", lower))
        best_score = 0
        best_fact = None
        for entry in learned:
            trigger_phrase = entry.get("trigger", "").strip().lower()
            if trigger_phrase and (trigger_phrase in lower or lower in trigger_phrase):
                best_score = 100
                best_fact = entry
                break
            triggers = {w.lower() for w in entry.get("trigger_keywords", [])}
            if not triggers:
                continue
            overlap = len(triggers & prompt_words)
            if overlap > 0 and overlap > best_score:
                best_score = overlap
                best_fact = entry
        if best_fact and (best_score >= 100 or best_score >= 2 or (len(best_fact.get("trigger_keywords", [])) == 1 and best_score >= 1)):
            # Increment hit count
            best_fact["hit_count"] = best_fact.get("hit_count", 0) + 1
            try:
                memory.save()
            except Exception:
                pass
            return KernelEvaluation(
                can_answer_locally=True,
                confidence=0.92 if best_score >= 100 else 0.88,
                local_response=f"[Learned Rule] {best_fact['fact']}",
                recommended_tier="aivco-kernel",
                reason="Learned-facts / taught-rule cache match.",
            ).__dict__

    # ---- Subsystem 3: Arbiter — select optimal tier ----
    task_type, recommended_tier = _arbiter_select_tier(prompt)

    return KernelEvaluation(
        can_answer_locally=False,
        confidence=0.0,
        local_response=None,
        recommended_tier=recommended_tier,
        reason=f"Query requires LLM. Arbiter selected {recommended_tier} (task_type={task_type}).",
        task_type=task_type,
    ).__dict__


def _arbiter_select_tier(prompt: str) -> tuple[str, str]:
    """
    Subsystem 3: Autonomous Model Arbiter.
    Returns (task_type, recommended_tier) based on query anatomy.
    """
    lower = prompt.lower()

    # Tier 1 triggers: sensitive/personal content
    if any(kw in lower for kw in _SENSITIVE_PROMPT_TRIGGERS):
        return "private", "tier1"

    # Tier 3 triggers: deep reasoning or long prompts
    if len(prompt) > 350:
        return "deep", "tier3"
    if any(kw in lower for kw in _DEEP_REASONING_TRIGGERS):
        return "deep", "tier3"

    # Tier 2 default: fast cloud route
    return "general", "tier2"


# ---------------------------------------------------------------------------
# Subsystem 2 — Continuous Learning & Auto-Memory Extraction
# ---------------------------------------------------------------------------

# Patterns for detecting user self-teaching statements in chat
_PREF_REMEMBER = re.compile(
    r"\b(remember\s+that|always\s+use|never\s+(?:do|use)|my\s+favorite\s+"
    r"|i\s+prefer|i\s+like|i\s+love|i\s+hate|my\s+preference)\b",
    re.IGNORECASE,
)

_STACK_LEARNING = re.compile(
    r"\bi\s+(am\s+learning|use|started\s+using|switched\s+to|now\s+using)\s+(\w[\w.\-/]+)",
    re.IGNORECASE,
)

_FACT_TEACH = re.compile(
    r"\b(note\s+that|please\s+note|remember\s+that|fyi[,:]?|just\s+so\s+you\s+know)\s+(.+)",
    re.IGNORECASE,
)

_FINISHED_TASK = re.compile(
    r"\b(i\s+(?:just\s+)?finished|completed|done\s+with|shipped)\s+(.+)",
    re.IGNORECASE,
)

_REMIND_TASK = re.compile(
    r"\b(remind\s+me\s+to|i\s+need\s+to|todo[:\s]+|add\s+(?:a\s+)?task[:\s]+)\s*(.+)",
    re.IGNORECASE,
)


def learn_from_turn(
    user_prompt: str,
    assistant_response: str,
    memory: "MultiLayerMemory",
) -> list[str]:
    """
    Subsystem 2: Continuously extract implicit facts/preferences/tasks
    from each conversation turn and persist them to memory.

    Returns a list of extraction notes (for debug/transparency logging).
    Always idempotent and non-destructive.
    """
    extractions: list[str] = []
    prompt = user_prompt.strip()

    # Check for operator directives during turn learning
    dir_res = _intercept_operator_directives(prompt, memory)
    if dir_res:
        extractions.append(dir_res)

    # ---- Preference / rule learning ----
    if _PREF_REMEMBER.search(prompt):
        # Store full statement as a preference fact
        _upsert_learned_fact(
            memory=memory,
            keywords=re.findall(r"\b\w{4,}\b", prompt.lower()),
            fact=prompt,
            fact_type="preference",
        )
        extractions.append(f"[Learned preference] {prompt[:80]}")

    # ---- Stack / tool learning ----
    m = _STACK_LEARNING.search(prompt)
    if m:
        tool = m.group(2).strip()
        action = m.group(1).strip()
        note = f"User {action} {tool}"
        _upsert_learned_fact(
            memory=memory,
            keywords=[tool.lower(), action.lower().split()[-1]],
            fact=note,
            fact_type="stack",
        )
        extractions.append(f"[Learned tool] {note}")

    # ---- Direct fact teaching ----
    m = _FACT_TEACH.search(prompt)
    if m:
        fact_text = m.group(2).strip().rstrip(".")
        if len(fact_text) > 5:
            _upsert_learned_fact(
                memory=memory,
                keywords=re.findall(r"\b\w{4,}\b", fact_text.lower())[:6],
                fact=fact_text,
                fact_type="taught_fact",
            )
            extractions.append(f"[Learned fact] {fact_text[:80]}")

    # ---- Finished task signals ----
    m = _FINISHED_TASK.search(prompt)
    if m:
        task_desc = m.group(2).strip().rstrip(".")
        active_id = memory.get_active_project()
        if active_id and task_desc:
            try:
                memory.add_episode(
                    event=f"Completed: {task_desc}",
                    outcome="logged-auto",
                    project_id=active_id,
                )
                extractions.append(f"[Auto-episode] Completed: {task_desc}")
            except Exception:
                pass

    # ---- Reminder / todo signals ----
    m = _REMIND_TASK.search(prompt)
    if m:
        task_desc = m.group(2).strip().rstrip(".")
        active_id = memory.get_active_project()
        if active_id and task_desc:
            try:
                memory.update_project_task(active_id, task_desc, mark_completed=False)
                extractions.append(f"[Auto-task] Added: {task_desc}")
            except Exception:
                pass

    return extractions


def _upsert_learned_fact(
    memory: "MultiLayerMemory",
    keywords: list[str],
    fact: str,
    fact_type: str,
) -> None:
    """
    Store or update a learned fact in memory.learned_facts list.
    Deduplicates by normalized fact content.
    """
    if not hasattr(memory, "learned_facts"):
        memory.learned_facts: list[dict] = []

    normalized = fact.strip().lower()
    for entry in memory.learned_facts:
        if entry.get("fact", "").strip().lower() == normalized:
            entry["hit_count"] = entry.get("hit_count", 0) + 1
            return  # already exists

    memory.learned_facts.append({
        "trigger_keywords": [k for k in keywords if k][:8],
        "fact": fact.strip(),
        "fact_type": fact_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hit_count": 0,
    })
    try:
        memory.save()
    except Exception:
        pass

