"""
AIVCO FastAPI GUI Bridge Server — Argon Edition
Provides REST endpoints connecting the browser GUI dashboard to
AIVCO's Cognitive Kernel, MultiLayerMemory, CommandController, and LLMController.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Optional, Dict, Any, List
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from memory.multi_layer_memory import MultiLayerMemory, CoreFact
from memory.memory_controller import MemoryController
from memory.dataset_logger import FineTuneDatasetLogger
from command_controller import CommandController
from llm_controller import LLMController, ModelUnavailableError, ModelGenerationError
from cognitive_kernel import (
    evaluate_query,
    learn_from_turn,
    scrub_sensitive_context,
    detect_explicit_model_and_intent,
    KernelCognitiveArbiter,
    build_executive_envelope,
    synthesize_elite_prompt,
    extract_fuzzy_provider_directive,
)
from browser_agent import browser_agent, search_web, universal_web_agent

BASE_DIR = Path(__file__).resolve().parent
V2_MEMORY_FILE = str(BASE_DIR / "data" / "v2_memory.json")
DATASET_FILE = str(BASE_DIR / "data" / "fine_tune_dataset.jsonl")
INDEX_HTML = BASE_DIR / "index.html"

# Initialize singletons
tiered_memory = MultiLayerMemory(storage_path=V2_MEMORY_FILE)
# V1 memory for CommandController backward compatibility
v1_memory_file = BASE_DIR / "memory" / "user" / "profile.json"
v1_memory = MemoryController(memory_path=v1_memory_file)
command_controller = CommandController(memory=v1_memory, tiered_memory=tiered_memory)
llm_controller = LLMController()
dataset_logger = FineTuneDatasetLogger(filepath=DATASET_FILE)

# Stateful Session Model Lock
active_model_override: Optional[str] = None

app = FastAPI(title="AIVCO GUI Bridge — Argon Edition", version="2.0")

# Enable CORS for localhost and frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Models ---

class ChatRequest(BaseModel):
    prompt: str
    target_model: Optional[str] = "auto"
    tier: Optional[int] = 2
    auto_route: Optional[bool] = True
    attachments: Optional[List[Dict[str, Any]]] = None


class DatasetApproveRequest(BaseModel):
    prompt: Optional[str] = None
    response: Optional[str] = None
    source: Optional[str] = "approved_chat"
    task_type: Optional[str] = "chat"


class TaskAddRequest(BaseModel):
    text: str
    tag: Optional[str] = "High"
    project: Optional[str] = None


class TaskToggleRequest(BaseModel):
    id: Optional[int] = None
    text: str
    project: Optional[str] = None


class MemoryAddRequest(BaseModel):
    layer: str
    title: str
    body: str


class ProjectCreateRequest(BaseModel):
    name: str


class VaultLoginRequest(BaseModel):
    provider: str
    slot: Optional[int] = None


class VaultSetSlotRequest(BaseModel):
    slot: int


class VaultUpdateRequest(BaseModel):
    provider: str
    slot: Optional[int] = None
    email: Optional[str] = None
    password: Optional[str] = None


# --- Endpoints ---

@app.get("/")
def get_index():
    """Serve the Argon Edition HTML dashboard."""
    if INDEX_HTML.exists():
        return FileResponse(
            str(INDEX_HTML),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )
    return {
        "message": "AIVCO GUI Bridge is active. Place index.html in the project root to load the dashboard."
    }


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    """
    Returns active LLM tier readiness & kernel state for all 4 Brains.
    """
    local_ready = llm_controller.is_available()
    groq_key = bool(os.environ.get("GROQ_API_KEY", "").strip())
    gemini_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())

    return {
        "status": "healthy",
        "kernel": True,
        "tier1_local": {"status": "disabled (RAM optimized)", "ready": False},
        "tier2_groq": groq_key,
        "tier3_gemini": gemini_key,
        "brain4_web": True,
        "active_override": active_model_override,
        "tiers": {
            "local_llama": {
                "ready": False,
                "status": "disabled (RAM optimized)",
                "label": "Tier 1: Disabled (RAM Safe)",
            },
            "groq": {
                "ready": groq_key,
                "label": "Tier 2: Groq Cloud API",
            },
            "gemini": {
                "ready": gemini_key,
                "label": "Tier 3: Google Gemini",
            },
            "web_search": {
                "ready": True,
                "label": "Brain 4: Live Web Search Grounding",
            },
            "claude": {
                "ready": True,
                "label": "Claude 3.7 Sonnet (In-Chat Execution)",
            },
            "chatgpt": {
                "ready": True,
                "label": "ChatGPT 4o (In-Chat Execution)",
            },
            "grok": {
                "ready": True,
                "label": "Grok 3 (In-Chat Execution)",
            },
        },
    }


@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    """
    Hydrates full state for the Argon UI:
    - active_project: str
    - projects: dict of project_id -> list of [milestone_name, percent_complete]
    - layers: { "Semantic": [...], "Episodic": [...], "Procedural": [...], "Creative": [...] }
    - tasks: list of { id, text, tag, done, project }
    - dataset_stats: dict
    """
    active_id = tiered_memory.get_active_project() or "aivco"
    active_proj = tiered_memory.projects.get(active_id)

    # 1. Projects mapping with milestone percentages
    projects_map: Dict[str, List[Any]] = {}
    for pid, proj in tiered_memory.projects.items():
        milestones = []
        n_completed = len(proj.completed_milestones)
        n_pending = len(proj.pending_tasks)
        total = n_completed + n_pending

        # Completed milestones at 100%
        for idx, m in enumerate(proj.completed_milestones):
            milestones.append([m, 100])

        # Pending tasks with calculated progression estimates
        for idx, t in enumerate(proj.pending_tasks):
            pct = 25 if idx == 0 else 10
            milestones.append([t, pct])

        if not milestones:
            milestones = [["Project Initialization", 100]]

        projects_map[pid] = milestones

    # 2. Tasks mapping (both pending and completed)
    tasks_list: List[Dict[str, Any]] = []
    task_id_counter = 1

    # First add pending tasks
    for pid, proj in tiered_memory.projects.items():
        for t in proj.pending_tasks:
            # Tag heuristic
            tag = "Priority" if any(w in t.lower() for w in ("faiss", "vector", "router", "fix", "critical")) else "High"
            tasks_list.append({
                "id": task_id_counter,
                "text": t,
                "tag": tag,
                "done": False,
                "project": pid,
            })
            task_id_counter += 1

    # Add completed milestones
    for pid, proj in tiered_memory.projects.items():
        for m in proj.completed_milestones:
            tasks_list.append({
                "id": task_id_counter,
                "text": m,
                "tag": "Done",
                "done": True,
                "project": pid,
            })
            task_id_counter += 1

    # 3. Layers mapping (Semantic, Episodic, Procedural, Creative)
    semantic_items: List[Dict[str, Any]] = []
    procedural_items: List[Dict[str, Any]] = []
    creative_items: List[Dict[str, Any]] = []

    # Map core profile facts
    for k, fact in tiered_memory.core_profile.items():
        if isinstance(fact, CoreFact):
            val = fact.value
            time_str = getattr(fact, "updated_at", "")[:10] or "Permanent"
        else:
            val = str(fact)
            time_str = "Permanent"
        title = k.replace("_", " ").title()

        if any(w in k.lower() for w in ("tone", "rule", "protocol", "workflow", "constraint")):
            procedural_items.append({
                "id": f"proc_{k}",
                "title": title,
                "body": val,
                "tag": "Protocol",
                "time": time_str,
            })
        elif any(w in k.lower() for w in ("interest", "creative", "songwriting", "game")):
            creative_items.append({
                "id": f"crea_{k}",
                "title": title,
                "body": val,
                "tag": "Creative",
                "time": time_str,
            })
        else:
            semantic_items.append({
                "id": f"sem_{k}",
                "title": title,
                "body": val,
                "tag": "Core",
                "time": time_str,
            })

    # Map learned facts
    for idx, lf in enumerate(getattr(tiered_memory, "learned_facts", [])):
        ftype = lf.get("fact_type", "taught_rule")
        trig = lf.get("trigger", f"Rule #{idx+1}")
        fact_text = lf.get("fact", "")
        time_str = lf.get("timestamp", "")[:10]

        if ftype == "taught_rule" or "rule" in ftype:
            procedural_items.append({
                "id": f"lf_proc_{idx}",
                "title": f"Rule: {trig}",
                "body": fact_text,
                "tag": "Learned",
                "time": time_str,
            })
        else:
            semantic_items.append({
                "id": f"lf_sem_{idx}",
                "title": f"Fact: {trig}",
                "body": fact_text,
                "tag": "Learned",
                "time": time_str,
            })

    # Map episodes
    episodic_items: List[Dict[str, Any]] = []
    for ep in tiered_memory.get_recent_episodes(limit=25):
        date_str = ep.timestamp[:10] if ep.timestamp else "Recent"
        episodic_items.append({
            "id": ep.id,
            "title": ep.event[:60] if ep.event else "Episode",
            "body": f"{ep.event} (Outcome: {ep.outcome})",
            "tag": ep.project_id or "General",
            "time": date_str,
        })

    # 4. Dataset stats
    dstats = dataset_logger.stats()

    # Core profile summary with native boolean/string values for test compliance
    core_summary = {}
    for k, v in tiered_memory.core_profile.items():
        if isinstance(v, CoreFact):
            val = v.value
            if val.lower() == "true":
                core_summary[k] = True
            elif val.lower() == "false":
                core_summary[k] = False
            else:
                core_summary[k] = val
        elif isinstance(v, bool):
            core_summary[k] = v
        else:
            s_val = str(v)
            if s_val.lower() == "true":
                core_summary[k] = True
            elif s_val.lower() == "false":
                core_summary[k] = False
            else:
                core_summary[k] = s_val

    return {
        "active_project": active_id,
        "active_project_id": active_id,
        "projects": projects_map,
        "tasks": tasks_list,
        "layers": {
            "Semantic": semantic_items,
            "Episodic": episodic_items,
            "Procedural": procedural_items,
            "Creative": creative_items,
        },
        "dataset_stats": dstats,
        # Backward-compatibility fields:
        "completed_milestones": active_proj.completed_milestones if active_proj else [],
        "pending_tasks": active_proj.pending_tasks if active_proj else [],
        "episodes": [asdict(e) for e in tiered_memory.get_recent_episodes(limit=10)],
        "core_profile": core_summary,
        "core_profile_raw": {
            k: asdict(v) if isinstance(v, CoreFact) else {
                "key": k, "value": str(v), "importance": "high", "updated_at": "Permanent"
            } for k, v in tiered_memory.core_profile.items()
        },
    }


def is_web_result_failure(web_res: Optional[str]) -> bool:
    """Checks if web bridge execution failed or hit rate limit."""
    if not web_res or not web_res.strip():
        return True
    s = web_res.strip().lower()
    return (
        "rate_limit_reached" in s
        or "[aivco-kernel] chrome extension" in s
        or "execution timed out waiting for ai response" in s
        or "tab connection error" in s
        or "failed to acquire tab" in s
        or "rate limit" in s
    )


async def execute_web_failover(
    target_tier: str,
    clean_prompt: str,
    raw_prompt: str,
    handshake_msg: str,
    t0: float,
) -> Dict[str, Any]:
    """Cascades to Tier 2 Groq/Gemini when Brain 4 web provider fails or hits rate limit."""
    active_proj = tiered_memory.get_active_project()
    tiered_context = tiered_memory.assemble_context(
        active_project_id=active_proj,
        cloud_safe=True,
    )
    failover_notice = (
        f"⚡ [aivco-kernel] {target_tier.upper()} web session unavailable/rate-limited. "
        "Auto-cascaded to Tier 2 Engine:\n\n"
    )
    system_prompt = (
        "You are AIVCO, a concise, intelligent, practical personal AI assistant.\n\n"
        f"MEMORY CONTEXT:\n{tiered_context}\n\n"
        "Synthesize an accurate, direct, production-grade response. "
        "Follow all operator directives strictly."
    )
    elite_prompt = synthesize_elite_prompt(clean_prompt, "groq", tiered_context)
    try:
        gen_res = llm_controller.route_and_generate(
            system_prompt=system_prompt,
            user_prompt=elite_prompt,
            task_type="general",
            force_provider="groq",
        )
        resp_text = gen_res.get("response", "")
        used_prov = gen_res.get("provider_used", "groq")
    except Exception as exc:
        resp_text = f"Fallback error: {exc}"
        used_prov = "tier2_fallback"

    final_res = f"{failover_notice}{resp_text}"
    latency_ms = max(10, round((time.time() - t0) * 1000))
    learn_from_turn(raw_prompt, final_res, tiered_memory)
    dataset_logger.log(
        source=f"failover_{target_tier}_to_{used_prov}",
        task_type="web_agent_failover",
        system_content=system_prompt[:500],
        user_content=raw_prompt,
        assistant_content=final_res,
    )
    return {
        "response": final_res,
        "provider": f"tier2:cascaded-from-{target_tier}",
        "src": "Tier 2 (Failover)",
        "tier": 2,
        "lat": latency_ms,
        "web_searched": False,
        "handshake": handshake_msg,
    }


@app.post("/api/chat")
async def post_chat(req: ChatRequest) -> Dict[str, Any]:
    """
    Multi-Brain Processing Architecture:
    - Brain 1 ([aivco-kernel]): Stage 0/1 Zero-LLM resolver & directives (<5ms)
    - Brain 4B ([brain4:browser]): Browser Agent Automation (Claude, ChatGPT, Grok)
    - Brain 4A ([brain4:web-search]): Live Web Search Grounding (DuckDuckGo + Wikipedia)
    - Brain 2 & 3: Fast Cloud & Deep Reasoning (Groq, Gemini, Local 5B) with forced routing
    """
    global active_model_override
    raw_prompt = req.prompt.strip()
    if not raw_prompt:
        raise HTTPException(status_code=400, detail="Empty prompt provided.")

    # Unpack attachments if present
    if req.attachments:
        attachment_text = ""
        for att in req.attachments:
            name = att.get("name", "file.txt")
            content = att.get("content", "")
            attachment_text += f"\n\n[Attached File: {name}]\n```\n{content}\n```\n"
        if attachment_text:
            raw_prompt = f"{raw_prompt}{attachment_text}"

    t0 = time.time()

    # ── Stage 0: Sub-Second Kernel Cognitive Arbitration (< 5ms) ─────────────
    is_auto_mode = req.target_model in ("auto", "AUTO (Arbiter)", None)
    target_tier, handshake_msg = KernelCognitiveArbiter.arbitrate(
        raw_prompt,
        target_override=None if is_auto_mode else (req.target_model or active_model_override),
    )
    if handshake_msg:
        print(f"[KernelCognitiveArbiter] Arbitrated '{raw_prompt[:40]}' -> Tier: {target_tier} | Handshake: {handshake_msg}")

    # ── Stage 0A: Natural Language Model Detector & Switch ──────────────────
    detected = detect_explicit_model_and_intent(raw_prompt)

    if detected.get("is_model_switch_only"):
        active_model_override = detected["switch_provider"]
        latency_ms = max(1, round((time.time() - t0) * 1000))
        return {
            "response": f"Sir, default model locked to [{detected['switch_provider']}].",
            "provider": "aivco-kernel",
            "src": "Procedural",
            "tier": 0,
            "lat": latency_ms,
            "web_searched": False,
            "handshake": handshake_msg,
        }

    # ── Stage 0B: Desktop Browser Automation (Brain 4B) — Explicit Command ONLY ──
    if detected.get("is_desktop_browser_request"):
        b_target = detected.get("browser_target") or "google"
        clean_q = detected.get("clean_prompt") or raw_prompt
        dispatch_info = browser_agent.open_and_dispatch(b_target, clean_q)
        latency_ms = max(10, round((time.time() - t0) * 1000))
        resp_msg = f"[Brain 4 Browser Agent] Dispatched desktop browser window to {b_target.upper()} ({dispatch_info['url']})."
        return {
            "response": resp_msg,
            "provider": f"brain4:browser-{b_target}",
            "src": "Brain 4",
            "tier": 4,
            "lat": latency_ms,
            "web_searched": False,
            "handshake": handshake_msg,
        }

    # ── Stage 0C: CommandController handling ──────────────────────────────────
    cmd_result = command_controller.handle(raw_prompt)
    if cmd_result is not None:
        latency_ms = max(1, round((time.time() - t0) * 1000))
        return {
            "response": cmd_result,
            "src": "Procedural",
            "provider": "aivco-kernel",
            "tier": 0,
            "lat": latency_ms,
            "web_searched": False,
            "handshake": handshake_msg,
        }

    # ── Stage 0-Arbiter: Auto-Mode Kernel Direct Execution (<10ms) ───────────
    if is_auto_mode and not active_model_override and target_tier == "kernel":
        # Direct math evaluation (<2ms)
        m_calc = re.search(r"^(?:what is|calc)\s*(\d+\s*[\+\-\*\/]\s*\d+)", raw_prompt.strip(), re.IGNORECASE)
        if m_calc:
            expr = m_calc.group(1).replace(" ", "")
            m_ops = re.match(r"^(\d+)([\+\-\*\/])(\d+)$", expr)
            if m_ops:
                a, op, b = int(m_ops.group(1)), m_ops.group(2), int(m_ops.group(3))
                res = a + b if op == "+" else (a - b if op == "-" else (a * b if op == "*" else (round(a / b, 4) if b != 0 else "Error: Division by zero")))
                latency_ms = max(1, round((time.time() - t0) * 1000))
                return {
                    "response": f"[aivco-kernel] Calculation: {expr} = {res}",
                    "src": "Procedural",
                    "provider": "aivco-kernel",
                    "tier": 0,
                    "lat": latency_ms,
                    "web_searched": False,
                    "handshake": handshake_msg,
                }

        # Direct clear/clean instruction
        clean_lower = raw_prompt.strip().lower()
        if re.match(r"^(clear|reset|clean)\s+(screen|memory|chat|context)$", clean_lower):
            latency_ms = max(1, round((time.time() - t0) * 1000))
            return {
                "response": f"[aivco-kernel] Direct instruction executed: {clean_lower}.",
                "src": "Procedural",
                "provider": "aivco-kernel",
                "tier": 0,
                "lat": latency_ms,
                "web_searched": False,
                "handshake": handshake_msg,
            }

        # Zero-LLM local evaluation (identity, profile, tasks left, project status)
        kernel_eval = evaluate_query(raw_prompt, tiered_memory)
        if kernel_eval.get("can_answer_locally"):
            local_res = kernel_eval.get("local_response", "")
            learn_from_turn(raw_prompt, local_res, tiered_memory)
            dataset_logger.log(
                source="aivco_kernel",
                task_type="kernel",
                system_content="AIVCO Cognitive Kernel — Stage 0 zero-LLM local resolution.",
                user_content=raw_prompt,
                assistant_content=local_res,
            )
            latency_ms = max(1, round((time.time() - t0) * 1000))
            return {
                "response": local_res,
                "src": "Semantic",
                "provider": "aivco-kernel",
                "tier": 0,
                "lat": latency_ms,
                "web_searched": False,
                "handshake": handshake_msg,
            }

    # ── Stage 0-Arbiter: Auto-Mode Web AI Escalation (Brain 4 Claude/ChatGPT) ──
    if is_auto_mode and not active_model_override and target_tier in ("claude", "chatgpt", "grok", "gemini_web"):
        clean_prompt = detected.get("clean_prompt") or raw_prompt
        # Executive Context Broker: wrap prompt with executive brief and memory context
        active_proj = tiered_memory.get_active_project()
        tiered_context = tiered_memory.assemble_context(
            active_project_id=active_proj,
            cloud_safe=True,
        )
        prompt_with_envelope = build_executive_envelope(clean_prompt, target_tier, tiered_context)

        web_res = await universal_web_agent.dispatch_prompt_async(
            target_tier,
            prompt_with_envelope,
            timeout=180.0,
            auto_launch_wait=15.0,
        )
        if is_web_result_failure(web_res):
            return await execute_web_failover(target_tier, clean_prompt, raw_prompt, handshake_msg, t0)
        latency_ms = max(10, round((time.time() - t0) * 1000))
        learn_from_turn(raw_prompt, web_res, tiered_memory)
        dataset_logger.log(
            source=f"web_account_{target_tier}",
            task_type="web_agent",
            system_content=f"AIVCO Cognitive Arbiter -> Brain 4 ({target_tier}) execution.",
            user_content=raw_prompt,
            assistant_content=web_res,
        )
        return {
            "response": web_res,
            "provider": f"web-account:{target_tier}",
            "src": "Brain 4 (Web Agent)",
            "tier": 4,
            "lat": latency_ms,
            "web_searched": False,
            "handshake": handshake_msg,
        }

    # ── Stage 0D: Universal Web AI Accounts Execution (Brain 4B) ──────────────
    # Real personal web AI accounts (Claude, ChatGPT, Grok, Gemini Web) via Playwright DOM
    WEB_PROVIDERS = (
        "claude", "chatgpt", "grok", "gemini_web", "gemini-web",
        "claude_web", "chatgpt_web", "grok_web",
    )
    explicit_provider = (
        detected.get("target_provider")
        or (req.target_model if req.target_model not in ("auto", "web_search", "web") else None)
        or active_model_override
    )
    target_web = (
        explicit_provider if explicit_provider in WEB_PROVIDERS
        else (req.target_model if req.target_model in WEB_PROVIDERS else None)
    )

    if target_web:
        if target_web in ("gemini-web", "gemini_web"):
            target_web = "gemini_web"
        elif target_web in ("claude", "claude_web"):
            target_web = "claude"
        elif target_web in ("chatgpt", "chatgpt_web"):
            target_web = "chatgpt"
        elif target_web in ("grok", "grok_web"):
            target_web = "grok"

        clean_prompt = detected.get("clean_prompt") or raw_prompt
        prompt_to_send = clean_prompt
        is_web_grounded = bool(detected.get("run_web_search") or req.target_model in ("web_search", "web"))
        
        # If web search grounding was explicitly requested with a web account query
        if is_web_grounded:
            try:
                snippets = search_web(clean_prompt)
                if snippets:
                    prompt_to_send = (
                        f"[Grounding Information from Web Search]:\n{snippets}\n\n"
                        f"User Prompt: {clean_prompt}\n"
                        f"Please answer the user prompt accurately using the grounding information above."
                    )
            except Exception:
                pass

        active_proj = tiered_memory.get_active_project()
        tiered_context = tiered_memory.assemble_context(
            active_project_id=active_proj,
            cloud_safe=True,
        )
        prompt_to_send = build_executive_envelope(prompt_to_send, target_web, tiered_context)

        web_res = await universal_web_agent.dispatch_prompt_async(
            target_web,
            prompt_to_send,
            timeout=180.0,
            auto_launch_wait=15.0,
        )
        if is_web_result_failure(web_res):
            return await execute_web_failover(target_web, clean_prompt, raw_prompt, handshake_msg, t0)
        latency_ms = max(10, round((time.time() - t0) * 1000))
        learn_from_turn(raw_prompt, web_res, tiered_memory)
        dataset_logger.log(
            source=f"web_account_{target_web}",
            task_type="web_agent",
            system_content=f"AIVCO ExtensionBridgeRouter -> {target_web} Chrome Extension execution.",
            user_content=raw_prompt,
            assistant_content=web_res,
        )
        return {
            "response": web_res,
            "provider": f"web-account:{target_web}",
            "src": "Brain 4 (Web Agent)",
            "tier": 4,
            "lat": latency_ms,
            "web_searched": is_web_grounded,
            "handshake": handshake_msg,
        }

    # ── Stage 0E: Live Web Search Grounding for Cloud API Models (Brain 4A) ────
    is_web_search = bool(detected.get("run_web_search") or req.target_model in ("web_search", "web"))
    if is_web_search:
        clean_search_q = detected.get("clean_prompt") or raw_prompt
        web_context = search_web(clean_search_q)
        target_prov = (
            explicit_provider
            if (explicit_provider and explicit_provider not in ("auto", "web_search", "web") and explicit_provider not in WEB_PROVIDERS)
            else "gemini"
        )
        
        active_proj = tiered_memory.get_active_project()
        tiered_context = tiered_memory.assemble_context(
            active_project_id=active_proj,
            cloud_safe=True,
        )
        system_prompt = (
            "You are AIVCO, a concise, intelligent, practical personal AI assistant.\n\n"
            f"{web_context}\n\n"
            f"MEMORY CONTEXT:\n{tiered_context}\n\n"
            "Synthesize an accurate, direct answer grounded in the web context with references and citations. "
            "Follow all operator directives strictly."
        )
        try:
            gen_res = llm_controller.route_and_generate(
                system_prompt=system_prompt,
                user_prompt=clean_search_q,
                task_type="deep" if target_prov == "gemini" else "general",
                force_provider=target_prov,
            )
            resp_text = gen_res.get("response", "")
            used_prov = gen_res.get("provider_used", target_prov)
        except Exception:
            # Fallback to direct web grounding synthesis when API keys are not configured in .env
            resp_text = (
                f"[AIVCO Real-Time Web Grounding]:\n\n"
                f"{web_context}\n\n"
                f"(Grounded directly from live web search — DuckDuckGo & Wikipedia verified knowledge)."
            )
            used_prov = f"web_grounding:{target_prov}"
        
        latency_ms = max(10, round((time.time() - t0) * 1000))
        
        learn_from_turn(raw_prompt, resp_text, tiered_memory)
        dataset_logger.log(
            source=f"web_search_{used_prov}",
            task_type="web_search",
            system_content=system_prompt[:500],
            user_content=raw_prompt,
            assistant_content=resp_text,
        )
        prov_label = used_prov
        if "gemini" in used_prov and not used_prov.startswith("tier"):
            prov_label = f"tier3:{used_prov}"
        elif "groq" in used_prov and not used_prov.startswith("tier"):
            prov_label = f"tier2:{used_prov}"

        return {
            "response": resp_text,
            "provider": prov_label,
            "src": "Brain 4 (Web Grounding)",
            "tier": 3 if "gemini" in used_prov else 2,
            "lat": latency_ms,
            "web_searched": True,
            "handshake": handshake_msg,
        }

    # ── Stage 1: CognitiveKernel Zero-LLM Evaluation ─────────────────────────
    kernel_eval = evaluate_query(raw_prompt, tiered_memory)

    if kernel_eval.get("can_answer_locally") and kernel_eval.get("confidence", 0) >= 0.85:
        is_directive = "directive" in kernel_eval.get("reason", "").lower()
        is_who_am_i = bool(re.search(r"\b(who\s+am\s+i)\b", raw_prompt, re.IGNORECASE))
        if not explicit_provider or explicit_provider == "aivco-kernel" or is_directive or is_who_am_i:
            local_res = kernel_eval.get("local_response", "")
            learn_from_turn(raw_prompt, local_res, tiered_memory)
            dataset_logger.log(
                source="aivco_kernel",
                task_type="kernel",
                system_content="AIVCO Cognitive Kernel — Stage 1 zero-LLM local resolution.",
                user_content=raw_prompt,
                assistant_content=local_res,
            )
            latency_ms = max(2, round((time.time() - t0) * 1000))
            return {
                "response": local_res,
                "src": "Semantic",
                "provider": "aivco-kernel",
                "tier": 0,
                "lat": latency_ms,
                "web_searched": False,
                "handshake": handshake_msg,
            }

    # ── Stage 2: Autonomous LLM Escalation ────────────────────────────────────
    clean_prompt = detected.get("clean_prompt") or raw_prompt
    lower_prompt = clean_prompt.lower()

    # Automatic Web Agent fallback when cloud API keys are commented out in .env
    has_cloud_keys = bool(os.environ.get("GROQ_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip())
    if not has_cloud_keys and explicit_provider not in ("local", "local_llama"):
        fallback_target = explicit_provider or "claude"
        active_proj = tiered_memory.get_active_project()
        tiered_context = tiered_memory.assemble_context(
            active_project_id=active_proj,
            cloud_safe=True,
        )
        enveloped_fallback = build_executive_envelope(clean_prompt, fallback_target, tiered_context)
        web_res = universal_web_agent.query(fallback_target, enveloped_fallback)
        latency_ms = max(10, round((time.time() - t0) * 1000))
        learn_from_turn(raw_prompt, web_res, tiered_memory)
        dataset_logger.log(
            source=f"web_agent_{fallback_target}",
            task_type="web_agent_fallback",
            system_content=f"AIVCO UniversalWebAgent -> {fallback_target} web fallback.",
            user_content=raw_prompt,
            assistant_content=web_res,
        )
        return {
            "response": web_res,
            "provider": f"web-account:{fallback_target}",
            "src": "Brain 4 (Web Agent)",
            "tier": 4,
            "lat": latency_ms,
            "web_searched": False,
            "handshake": handshake_msg,
        }

    # Priority routing resolution
    force_provider = explicit_provider
    if force_provider == "gemini":
        task_type = "deep"
    elif force_provider == "groq":
        task_type = "general"
    elif force_provider in ("local", "local_llama"):
        task_type = "private"
    elif lower_prompt.startswith("deep:"):
        task_type = "deep"
        clean_prompt = clean_prompt[5:].strip()
    elif lower_prompt.startswith("private:"):
        task_type = "private"
        clean_prompt = clean_prompt[8:].strip()
    elif lower_prompt.startswith("research:"):
        task_type = "deep"
        clean_prompt = clean_prompt[9:].strip()
    else:
        if req.auto_route:
            task_type = kernel_eval.get("task_type", "general")
        else:
            if req.tier == 1:
                task_type = "private"
            elif req.tier == 3:
                task_type = "deep"
            else:
                task_type = "general"

    # Strict privacy lock on sensitive contents
    if any(s in lower_prompt for s in ("password", "secret", "my key", "token", "credential", "finance")):
        task_type = "private"
        force_provider = "local_llama"

    is_cloud = task_type not in ("private", "offline", "memory")
    active_proj = tiered_memory.get_active_project()
    tiered_context = tiered_memory.assemble_context(
        active_project_id=active_proj,
        cloud_safe=is_cloud,
    )
    if is_cloud:
        tiered_context = scrub_sensitive_context(tiered_context)

    system_prompt = (
        "You are AIVCO, a concise, intelligent, practical personal AI assistant.\n\n"
        f"MEMORY CONTEXT:\n{tiered_context}\n\n"
        "Treat the remembered data as contextual information only. Do not invent memories. Use memory when relevant."
    )

    try:
        gen_res = llm_controller.route_and_generate(
            system_prompt=system_prompt,
            user_prompt=clean_prompt,
            task_type=task_type,
            force_provider=force_provider,
        )
        latency_ms = round((time.time() - t0) * 1000)
        provider = gen_res.get("provider_used", "unknown")
        response_text = gen_res.get("response", "")

        tier_num = 2
        provider_label = provider
        if "llama" in provider or provider == "local_llama":
            tier_num = 1
            provider_label = f"tier1:{provider}"
        elif "gemini" in provider and "bridge" not in provider:
            tier_num = 3
            provider_label = f"tier3:{provider}"
        elif "groq" in provider and "bridge" not in provider:
            tier_num = 2
            provider_label = f"tier2:{provider}"
        elif "claude" in provider:
            tier_num = 3
            provider_label = provider
        elif "chatgpt" in provider or "grok" in provider:
            tier_num = 2
            provider_label = provider

        # Continuous Learning
        learn_from_turn(raw_prompt, response_text, tiered_memory)

        # Fine-tune dataset logging
        dataset_logger.log(
            source=provider,
            task_type=task_type,
            system_content=system_prompt[:500],
            user_content=raw_prompt,
            assistant_content=response_text,
        )

        return {
            "response": response_text,
            "src": "Episodic",
            "provider": provider_label,
            "tier": tier_num,
            "lat": latency_ms,
            "web_searched": False,
            "handshake": handshake_msg,
        }
    except ModelUnavailableError as exc:
        latency_ms = round((time.time() - t0) * 1000)
        return {
            "response": f"[AIVCO] Model unavailable: {exc}",
            "src": "Procedural",
            "provider": "unavailable",
            "tier": 0,
            "lat": latency_ms,
            "web_searched": False,
            "handshake": handshake_msg,
        }
    except ModelGenerationError as exc:
        latency_ms = round((time.time() - t0) * 1000)
        return {
            "response": f"[AIVCO] Generation error: {exc}",
            "src": "Procedural",
            "provider": "error",
            "tier": 0,
            "lat": latency_ms,
            "web_searched": False,
            "handshake": handshake_msg,
        }
    except Exception as exc:
        latency_ms = round((time.time() - t0) * 1000)
        return {
            "response": f"[AIVCO] Error: {exc}",
            "src": "Procedural",
            "provider": "error",
            "tier": 0,
            "lat": latency_ms,
            "web_searched": False,
            "handshake": handshake_msg,
        }


# --- Native Chrome Extension Bridge (WebSocket & REST) ---

@app.websocket("/ws/bridge")
async def websocket_bridge_endpoint(websocket: WebSocket):
    """
    WebSocket bridge server for AIVCO Native Chrome Extension (Manifest V3).
    Receives registrations, status heartbeats, and dispatches prompts directly
    to authenticated web AI tabs.
    """
    await websocket.accept()
    universal_web_agent.register_client(websocket)
    print("[bridge] Chrome Extension connected successfully.")
    try:
        while True:
            text_data = await websocket.receive_text()
            try:
                data = json.loads(text_data)
                action = data.get("action") or data.get("type")
                if action == "ping":
                    await websocket.send_text(json.dumps({"action": "pong"}))
                elif action == "progress":
                    req_id = data.get("request_id")
                    await websocket.send_text(json.dumps({"action": "pong", "request_id": req_id}))
                universal_web_agent.handle_incoming_message(data)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        universal_web_agent.unregister_client(websocket)
        print("[bridge] Chrome Extension disconnected.")
    except Exception as exc:
        universal_web_agent.unregister_client(websocket)
        print(f"[bridge] Chrome Extension WebSocket closed: {exc}")


@app.get("/api/browser/extension_status")
def get_extension_status() -> Dict[str, Any]:
    """Reports status of the Chrome Extension WebSocket bridge."""
    return universal_web_agent.status()


@app.post("/api/bridge/result")
async def receive_bridge_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Direct HTTP completion endpoint for AIVCO Native Chrome Extension.
    Allows background.js and content.js to resolve pending prompt dispatches
    instantly via HTTP, immune to WebSocket drops and channel timeouts.
    """
    request_id = payload.get("request_id")
    markdown = payload.get("markdown", "")
    error = payload.get("error")
    resolved = universal_web_agent.resolve_request(request_id, markdown=markdown, error=error)
    return {"status": "ok", "resolved": resolved}


@app.post("/api/browser/launch_cdp")
def launch_cdp_endpoint() -> Dict[str, Any]:
    """Deprecated: CDP mode migrated to AIVCO Native Chrome Extension."""
    return {
        "ok": True,
        "message": "CDP mode has been migrated to AIVCO Chrome Extension (Manifest V3). Load unpacked extension in chrome://extensions.",
        "connected": universal_web_agent.is_connected(),
    }


@app.get("/api/browser/cdp_status")
def cdp_status_endpoint() -> Dict[str, Any]:
    """Returns extension connection status for backward compatibility."""
    connected = universal_web_agent.is_connected()
    return {
        "running": connected,
        "connected": connected,
        "tabs": universal_web_agent.get_connected_tabs(),
        "status": "connected" if connected else "offline",
    }


@app.post("/api/browser/open_sessions")
def open_browser_sessions() -> Dict[str, Any]:
    """Compatibility endpoint."""
    return {
        "ok": True,
        "message": "Please open claude.ai, chatgpt.com, gemini.google.com, or grok.com directly in Chrome with the AIVCO extension loaded.",
    }


@app.get("/api/browser/status")
def get_browser_status() -> Dict[str, Any]:
    """Reports status of the Chrome Extension bridge."""
    return universal_web_agent.status()


# --- Dedicated AI Credentials Vault Endpoints ---

@app.get("/api/vault/status")
def get_vault_status_endpoint() -> Dict[str, Any]:
    """Returns the dedicated credentials vault status with all 20 Claude slots and 3 providers."""
    from browser_agent import get_vault_status_summary
    return get_vault_status_summary()


@app.post("/api/vault/login")
def vault_login_endpoint(req: VaultLoginRequest) -> Dict[str, Any]:
    """Launches the persistent agent browser on port 9222 (CDP Bridge)."""
    return universal_web_agent.launch_cdp()


@app.post("/api/vault/set_active_slot")
def set_active_slot_endpoint(req: VaultSetSlotRequest) -> Dict[str, Any]:
    """Switches the active Claude slot in the credentials vault."""
    from browser_agent import set_active_claude_slot
    return set_active_claude_slot(req.slot)


@app.post("/api/vault/update_credentials")
def update_credentials_endpoint(req: VaultUpdateRequest) -> Dict[str, Any]:
    """Deprecated: Strict credential isolation policy requires headed browser login only."""
    return {
        "ok": True,
        "message": "Direct credential storage is disabled. Please use manual headed browser login via 'Login' button.",
    }


# --- Interactive Mutation Endpoints ---

@app.post("/api/tasks/add")
def add_task(req: TaskAddRequest) -> Dict[str, Any]:
    """Add a pending task to target project (or active project)."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Task text cannot be empty.")

    target_pid = req.project or tiered_memory.get_active_project() or "aivco"
    if target_pid not in tiered_memory.projects:
        tiered_memory.register_project(target_pid, target_pid.upper())

    tiered_memory.update_project_task(target_pid, text, mark_completed=False)
    return {"ok": True, "project": target_pid, "text": text}


@app.post("/api/tasks/toggle")
def toggle_task(req: TaskToggleRequest) -> Dict[str, Any]:
    """Toggle a task between pending and completed."""
    target_text = req.text.strip()
    target_pid = req.project or tiered_memory.get_active_project() or "aivco"
    proj = tiered_memory.projects.get(target_pid)

    if not proj:
        # Check all projects for the task
        for pid, p in tiered_memory.projects.items():
            if target_text in p.pending_tasks or target_text in p.completed_milestones:
                target_pid = pid
                proj = p
                break

    if not proj:
        raise HTTPException(status_code=404, detail="Task not found in any project.")

    if target_text in proj.pending_tasks:
        # Mark done
        tiered_memory.update_project_task(target_pid, target_text, mark_completed=True)
        return {"ok": True, "done": True, "text": target_text}
    elif target_text in proj.completed_milestones:
        # Move back to pending
        proj.completed_milestones.remove(target_text)
        if target_text not in proj.pending_tasks:
            proj.pending_tasks.append(target_text)
        tiered_memory.save()
        return {"ok": True, "done": False, "text": target_text}

    # Fallback: fuzzy match by substring
    for t in list(proj.pending_tasks):
        if target_text.lower() in t.lower():
            tiered_memory.update_project_task(target_pid, t, mark_completed=True)
            return {"ok": True, "done": True, "text": t}

    for m in list(proj.completed_milestones):
        if target_text.lower() in m.lower():
            proj.completed_milestones.remove(m)
            proj.pending_tasks.append(m)
            tiered_memory.save()
            return {"ok": True, "done": False, "text": m}

    raise HTTPException(status_code=404, detail="Task text not found in project.")


@app.post("/api/memory/add")
def add_memory(req: MemoryAddRequest) -> Dict[str, Any]:
    """Commit a new memory engram into MultiLayerMemory."""
    layer = req.layer.strip().title()
    title = req.title.strip()
    body = req.body.strip()

    if not title and not body:
        raise HTTPException(status_code=400, detail="Title and body cannot both be empty.")

    active_id = tiered_memory.get_active_project()

    if layer == "Episodic":
        ep = tiered_memory.add_episode(
            event=f"{title}: {body}" if title and body else (title or body),
            outcome="logged",
            project_id=active_id,
        )
        return {"ok": True, "id": ep.id, "layer": "Episodic"}

    # For Semantic, Procedural, Creative:
    # Save as a learned rule / fact so it's searchable and retrievable
    if not hasattr(tiered_memory, "learned_facts"):
        tiered_memory.learned_facts = []

    clean_trigger = title or body[:30]
    keywords = [k for k in re.findall(r"\b\w+\b", clean_trigger.lower()) if len(k) >= 3]

    entry = {
        "trigger": clean_trigger,
        "trigger_keywords": keywords,
        "fact": body or title,
        "fact_type": layer.lower(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hit_count": 0,
    }
    tiered_memory.learned_facts.append(entry)

    # Also save to core profile if it's a clean key
    clean_key = re.sub(r"[^a-z0-9_]", "_", title.lower()).strip("_")
    if clean_key and len(clean_key) <= 40 and " " not in clean_key:
        tiered_memory.set_core_fact(clean_key, body or title)
    else:
        tiered_memory.save()

    return {"ok": True, "layer": layer, "title": clean_trigger}


@app.post("/api/projects/create")
def create_project(req: ProjectCreateRequest) -> Dict[str, Any]:
    """Create a new project and switch active project to it."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name cannot be empty.")

    pid = re.sub(r"[^a-z0-9\-]", "-", name.lower()).strip("-")
    if not pid:
        pid = f"proj-{int(time.time())}"

    tiered_memory.register_project(project_id=pid, title=name)
    return {"ok": True, "project_id": pid, "title": name}


@app.post("/api/dataset/approve")
def approve_dataset(req: DatasetApproveRequest) -> Dict[str, Any]:
    """Logs a user-approved conversation pair to the fine-tune dataset."""
    if not req.prompt or not req.response:
        raise HTTPException(status_code=400, detail="Prompt and response are required.")
    dataset_logger.log(
        source=req.source or "approved_chat",
        task_type=req.task_type or "chat",
        system_content="AIVCO Golden Dataset — User Approved Response Pair.",
        user_content=req.prompt,
        assistant_content=req.response,
    )
    return {"ok": True, "total": dataset_logger.stats()["total_entries"]}
