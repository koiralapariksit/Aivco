"""
AIVCO LLM Controller

Thin wrapper around the llama.cpp CLI and multi-tier LLM routing.
Supports:
- Tier 1: Local offline llama-cli execution (strict privacy lock when requested)
- Tier 2: Groq Cloud API (Fast 70B Engine)
- Tier 3: Gemini API (Deep Analysis & Large Context)
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import urllib.error
import urllib.request

LLAMA_PATH = r"C:\Users\koira\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-cli.exe"

MODEL_PATH = r"C:\Users\koira\.cache\huggingface\hub\models--ggml-org--Qwen3-0.6B-GGUF\snapshots\b5f37287796e5be0ea3dab2e7430873fb3f73e49\Qwen3-0.6B-Q4_0.gguf"
LLM_TIMEOUT_SECONDS = 180
LLM_MAX_TOKENS = 512


class AIVCOModelError(RuntimeError):
    """Base exception for expected local or remote model failures."""


class ModelUnavailableError(AIVCOModelError):
    """The model executable, model file, or API credentials are missing."""


class ModelGenerationError(AIVCOModelError):
    """The model failed during generation or network request."""


EXECUTIVE_AUTONOMY_MANDATE = (
    "EXECUTIVE AUTONOMY MANDATE (ZERO HAND-HOLDING):\n"
    "1. NEVER ask the operator clarifying questions (e.g., never ask 'which framework?', 'what platform?', or 'would you like starter code?').\n"
    "2. NEVER reply with a high-level text roadmap, folder tree scaffold, or architecture outline when the operator asks to build, make, or clone an app, website, or UI.\n"
    "3. Act as an autonomous, decisive CTO/CEO: immediately choose the best single-file HTML5/CSS3/JS architecture and output the COMPLETE, fully functional, self-contained `<!DOCTYPE html>` application inside a single ```html code block so it renders immediately in AIVCO's Live Preview drawer.\n"
    "4. Even if the operator writes typos (e.g., 'claong website of nelflix', 'buikd', 'amek'), infer the exact target product and build the complete working clone on the first turn."
)

ANTI_HALLUCINATION_MANDATE = (
    "ANTI-HALLUCINATION & TEMPORAL VERIFICATION MANDATE:\n"
    "If a user asks about real-time, current, or post-2024 events (sports scores, match results, transfers, news) "
    "and you do not have live web access or grounded search results, you MUST explicitly state: "
    "'I do not have live web access or real-time verification for post-2024 events.' "
    "NEVER fabricate, guess, or invent sports matches, scores, player transfers, or news headlines."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are AIVCO, a concise, intelligent, practical personal AI assistant.\n\n"
    f"{EXECUTIVE_AUTONOMY_MANDATE}\n\n"
    f"{ANTI_HALLUCINATION_MANDATE}"
)


def _is_web_failure(web_res: Optional[str]) -> bool:
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


def load_env_file(env_path: str = ".env") -> None:
    """
    Zero-dependency .env loader.
    Resolves relative path to project root if default '.env' is passed.
    Supports utf-8-sig (BOM) and utf-16 fallbacks for Windows Notepad.
    Auto-detects raw API keys (gsk_ -> GROQ_API_KEY, AIza -> GEMINI_API_KEY).
    Maps GOOGLE_API_KEY alias to GEMINI_API_KEY.
    """
    path_obj = Path(env_path)
    if env_path == ".env" and not path_obj.is_absolute():
        path_obj = Path(__file__).resolve().parent.parent / ".env"

    if not path_obj.exists():
        return

    raw_bytes = None
    try:
        raw_bytes = path_obj.read_bytes()
    except OSError:
        return

    content = ""
    # Try encodings: utf-8-sig (handles UTF-8 with and without BOM), utf-16, latin1
    for enc in ("utf-8-sig", "utf-16", "utf-8", "latin1"):
        try:
            content = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]

            os.environ.setdefault(key, val)
            if key == "GOOGLE_API_KEY":
                os.environ.setdefault("GEMINI_API_KEY", val)
        else:
            # Smart Key Detection (Safe Auto-Mapping for raw pasted keys)
            clean_val = line.strip().strip('"').strip("'")
            if clean_val.startswith("gsk_"):
                os.environ.setdefault("GROQ_API_KEY", clean_val)
            elif clean_val.startswith("AIza"):
                os.environ.setdefault("GEMINI_API_KEY", clean_val)
            elif clean_val.startswith("sk-ant-"):
                os.environ.setdefault("ANTHROPIC_API_KEY", clean_val)
            elif clean_val.startswith("sk-or-"):
                os.environ.setdefault("OPENROUTER_API_KEY", clean_val)
            elif clean_val.startswith("xai-"):
                os.environ.setdefault("XAI_API_KEY", clean_val)


class LLMController:
    local_enabled: bool = False

    def __init__(
        self,
        llama_path: Optional[str] = None,
        model_path: Optional[str] = None,
        env_path: str = ".env",
    ):
        self.local_enabled = False
        self.env_path = env_path
        load_env_file(self.env_path)
        # Brain 2 (Local LLM) hard-killed to prevent OS crashes & RAM exhaustion
        self.llama_path = None
        self.model_path = None

    def is_available(self) -> bool:
        return False

    def _generate_local(self, system_prompt: str = "", user_prompt: str = "") -> str:
        return "[aivco-kernel] Local LLM tier permanently disabled to conserve system RAM."

    def generate(self, system_prompt: str = "", user_prompt: str = "") -> str:
        """
        Local LLM generation permanently disabled to conserve system RAM.
        Raises ModelUnavailableError so callers know local LLM cannot be run.
        """
        raise ModelUnavailableError("[aivco-kernel] Local LLM tier permanently disabled to conserve system RAM.")

    def _generate_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "openai/gpt-oss-120b",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ModelUnavailableError("GROQ_API_KEY is not configured.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AIVCO/2.0",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for turn in history:
                r = turn.get("role", "user")
                c = turn.get("content", "")
                if c:
                    messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": user_prompt})

        body = {
            "model": model,
            "messages": messages,
        }

        try:
            req_data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_text = resp.read().decode("utf-8")
                data = json.loads(resp_text)
                return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise ModelGenerationError(f"Groq generation failed: {exc}") from exc

    def _generate_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gemini-flash-latest",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ModelUnavailableError("GEMINI_API_KEY is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AIVCO/2.0",
        }
        contents = []
        if history:
            for turn in history:
                role = "user" if turn.get("role") == "user" else "model"
                c = turn.get("content", "")
                if c:
                    contents.append({
                        "role": role,
                        "parts": [{"text": c}],
                    })
        contents.append({
            "role": "user",
            "parts": [{"text": user_prompt}],
        })

        body = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": contents,
        }

        try:
            req_data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_text = resp.read().decode("utf-8")
                data = json.loads(resp_text)
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as exc:
            raise ModelGenerationError(f"Gemini generation failed: {exc}") from exc

    def _call_openai_compatible(
        self,
        endpoint_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: int = 40,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AIVCO/2.0",
        }
        if extra_headers:
            headers.update(extra_headers)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for turn in history:
                r = turn.get("role", "user")
                c = turn.get("content", "")
                if c:
                    messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": user_prompt})

        body = {"model": model, "messages": messages}
        try:
            req_data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(endpoint_url, data=req_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_text = resp.read().decode("utf-8")
                data = json.loads(resp_text)
                return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise ModelGenerationError(f"OpenAI-compatible request to {endpoint_url} failed: {exc}") from exc

    def _generate_claude(self, system_prompt: str, user_prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        res, _ = self._generate_claude_with_meta(system_prompt, user_prompt, history=history)
        return res

    def _generate_claude_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[str, str]:
        # 1. Anthropic Direct API
        ant_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        if ant_key:
            try:
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": ant_key.strip(),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                    "User-Agent": "AIVCO/2.0",
                }
                messages = []
                if history:
                    for turn in history:
                        r = turn.get("role", "user")
                        c = turn.get("content", "")
                        if c:
                            messages.append({"role": r, "content": c})
                messages.append({"role": "user", "content": user_prompt})

                body = {
                    "model": "claude-3-7-sonnet-20250219",
                    "max_tokens": 1500,
                    "system": system_prompt,
                    "messages": messages,
                }
                req_data = json.dumps(body).encode("utf-8")
                req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=40) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    text = "".join(part.get("text", "") for part in data.get("content", []))
                    if text:
                        return text.strip(), "claude-3-7-sonnet"
            except Exception:
                pass

        # 2. OpenRouter API
        or_key = os.environ.get("OPENROUTER_API_KEY")
        if or_key:
            try:
                res = self._call_openai_compatible(
                    "https://openrouter.ai/api/v1/chat/completions",
                    or_key.strip(),
                    "anthropic/claude-3.7-sonnet",
                    system_prompt,
                    user_prompt,
                    extra_headers={"HTTP-Referer": "https://aivco.org", "X-Title": "AIVCO"},
                    history=history,
                )
                if res:
                    return res, "openrouter:claude-3.7-sonnet"
            except Exception:
                pass

        # 3. Web Account DOM Automation via UniversalWebAgent
        from agents.browser_agent import universal_web_agent
        res = universal_web_agent.query("claude", user_prompt)
        return res, "web-account:claude"

    def _generate_chatgpt(self, system_prompt: str, user_prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        res, _ = self._generate_chatgpt_with_meta(system_prompt, user_prompt, history=history)
        return res

    def _generate_chatgpt_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[str, str]:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                res = self._call_openai_compatible(
                    "https://api.openai.com/v1/chat/completions",
                    openai_key.strip(),
                    "gpt-4o",
                    system_prompt,
                    user_prompt,
                    history=history,
                )
                if res:
                    return res, "gpt-4o"
            except Exception:
                pass

        or_key = os.environ.get("OPENROUTER_API_KEY")
        if or_key:
            try:
                res = self._call_openai_compatible(
                    "https://openrouter.ai/api/v1/chat/completions",
                    or_key.strip(),
                    "openai/gpt-4o",
                    system_prompt,
                    user_prompt,
                    extra_headers={"HTTP-Referer": "https://aivco.org", "X-Title": "AIVCO"},
                    history=history,
                )
                if res:
                    return res, "openrouter:gpt-4o"
            except Exception:
                pass

        # 3. Web Account DOM Automation via UniversalWebAgent
        from agents.browser_agent import universal_web_agent
        res = universal_web_agent.query("chatgpt", user_prompt)
        return res, "web-account:chatgpt"

    def _generate_grok(self, system_prompt: str, user_prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        res, _ = self._generate_grok_with_meta(system_prompt, user_prompt, history=history)
        return res

    def _generate_grok_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[str, str]:
        xai_key = os.environ.get("XAI_API_KEY")
        if xai_key:
            try:
                res = self._call_openai_compatible(
                    "https://api.x.ai/v1/chat/completions",
                    xai_key.strip(),
                    "grok-3",
                    system_prompt,
                    user_prompt,
                    history=history,
                )
                if res:
                    return res, "grok-3"
            except Exception:
                pass

        or_key = os.environ.get("OPENROUTER_API_KEY")
        if or_key:
            try:
                res = self._call_openai_compatible(
                    "https://openrouter.ai/api/v1/chat/completions",
                    or_key.strip(),
                    "x-ai/grok-3",
                    system_prompt,
                    user_prompt,
                    extra_headers={"HTTP-Referer": "https://aivco.org", "X-Title": "AIVCO"},
                    history=history,
                )
                if res:
                    return res, "openrouter:grok-3"
            except Exception:
                pass

        # 3. Web Account DOM Automation via UniversalWebAgent
        from agents.browser_agent import universal_web_agent
        res = universal_web_agent.query("grok", user_prompt)
        return res, "web-account:grok"

    def route_and_generate(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        task_type: str = "general",
        force_provider: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> dict:
        """
        Routes the prompt to the appropriate model tier with automatic fallback.
        Supports forced provider override (force_provider="claude", "chatgpt", "grok", "gemini", "groq", "local_llama").
        Returns:
            {"response": str, "provider_used": str, "fallback_triggered": bool}
        """
        if not system_prompt:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        # Rule 1: Forced Provider Selection
        effective_force = force_provider or (task_type if task_type in ("claude", "chatgpt", "grok", "gemini_web", "claude_web", "chatgpt_web", "grok_web", "gemini", "groq", "local_llama", "local") else None)

        def _invoke(fn, prov_name):
            if history is not None and prov_name in ("groq", "gemini", "claude", "chatgpt", "grok"):
                try:
                    return fn(system_prompt, user_prompt, history=history)
                except TypeError:
                    return fn(system_prompt, user_prompt)
            return fn(system_prompt, user_prompt)

        WEB_PROVIDERS = ("claude", "chatgpt", "grok", "gemini_web", "gemini-web", "claude_web", "chatgpt_web", "grok_web")
        if effective_force in WEB_PROVIDERS:
            target = "gemini_web" if effective_force in ("gemini_web", "gemini-web") else effective_force.replace("_web", "")
            from agents.browser_agent import universal_web_agent
            try:
                web_res = universal_web_agent.query(target, user_prompt)
                if _is_web_failure(web_res):
                    raise AIVCOModelError(f"Web provider {target} failed: {web_res}")
                return {
                    "response": web_res,
                    "provider_used": f"web-account:{target}",
                    "fallback_triggered": False,
                }
            except Exception:
                # Brain 4 Web failed / timed out / rate-limited: auto-cascade to Groq API -> Gemini API
                try:
                    response = _invoke(self._generate_groq, "groq")
                    return {
                        "response": response,
                        "provider_used": f"groq (fallback from {target})",
                        "fallback_triggered": True,
                    }
                except AIVCOModelError:
                    response = _invoke(self._generate_gemini, "gemini")
                    return {
                        "response": response,
                        "provider_used": f"gemini (fallback from {target})",
                        "fallback_triggered": True,
                    }

        if effective_force in ("local", "local_llama") or task_type in ("private", "memory", "offline"):
            try:
                response = self.generate(system_prompt, user_prompt)
                return {
                    "response": response,
                    "provider_used": "local_llama",
                    "fallback_triggered": False,
                }
            except AIVCOModelError as local_err:
                # Privacy block eliminated: operator requires cloud fallback (Groq -> Gemini) to execute
                try:
                    response = _invoke(self._generate_groq, "groq")
                    return {
                        "response": response,
                        "provider_used": "groq (fallback from local)",
                        "fallback_triggered": True,
                    }
                except AIVCOModelError as groq_err:
                    try:
                        response = _invoke(self._generate_gemini, "gemini")
                        return {
                            "response": response,
                            "provider_used": "gemini (fallback from local)",
                            "fallback_triggered": True,
                        }
                    except AIVCOModelError as gemini_err:
                        raise ModelUnavailableError(
                            f"All providers failed (local: {local_err}; groq: {groq_err}; gemini: {gemini_err})"
                        )

        if effective_force == "gemini":
            try:
                response = _invoke(self._generate_gemini, "gemini")
                return {
                    "response": response,
                    "provider_used": "gemini",
                    "fallback_triggered": False,
                }
            except AIVCOModelError:
                # Fallback to Groq if Gemini fails
                try:
                    response = _invoke(self._generate_groq, "groq")
                    return {
                        "response": response,
                        "provider_used": "groq (fallback from gemini)",
                        "fallback_triggered": True,
                    }
                except AIVCOModelError:
                    response = self.generate(system_prompt, user_prompt)
                    return {
                        "response": response,
                        "provider_used": "local_llama (fallback from gemini)",
                        "fallback_triggered": True,
                    }

        if effective_force == "groq":
            try:
                response = _invoke(self._generate_groq, "groq")
                return {
                    "response": response,
                    "provider_used": "groq",
                    "fallback_triggered": False,
                }
            except AIVCOModelError:
                # Fallback to Gemini if Groq fails
                try:
                    response = _invoke(self._generate_gemini, "gemini")
                    return {
                        "response": response,
                        "provider_used": "gemini (fallback from groq)",
                        "fallback_triggered": True,
                    }
                except AIVCOModelError:
                    response = self.generate(system_prompt, user_prompt)
                    return {
                        "response": response,
                        "provider_used": "local_llama (fallback from groq)",
                        "fallback_triggered": True,
                    }

        # Rule 2: Tier 3 Deep / Large Context
        if task_type in ("deep", "research") or (len(system_prompt) + len(user_prompt) > 6000):
            chain = [
                ("gemini", self._generate_gemini),
                ("groq", self._generate_groq),
                ("local_llama", self.generate),
            ]
        # Rule 3: Tier 2 Fast Default
        else:
            chain = [
                ("groq", self._generate_groq),
                ("gemini", self._generate_gemini),
                ("local_llama", self.generate),
            ]

        errors = []
        for idx, (provider_name, provider_fn) in enumerate(chain):
            try:
                response = _invoke(provider_fn, provider_name)
                return {
                    "response": response,
                    "provider_used": provider_name,
                    "fallback_triggered": idx > 0,
                }
            except AIVCOModelError as exc:
                errors.append((provider_name, exc))
                continue

        # If all providers failed:
        # Check if any cloud provider failed with ModelGenerationError (e.g. HTTP 400/401/403/429/etc.)
        cloud_generation_errors = [
            (name, err)
            for name, err in errors
            if name != "local_llama" and isinstance(err, ModelGenerationError)
        ]
        if cloud_generation_errors:
            err_details = "; ".join(f"[{name}] {str(err)}" for name, err in cloud_generation_errors)
            raise ModelGenerationError(f"Cloud model generation failed: {err_details}")

        # If the last error was recorded, raise it (e.g. ModelUnavailableError)
        if errors:
            raise errors[-1][1]
        raise ModelUnavailableError("No model provider available.")
