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
LLM_MAX_TOKENS = 32


class AIVCOModelError(RuntimeError):
    """Base exception for expected local or remote model failures."""


class ModelUnavailableError(AIVCOModelError):
    """The model executable, model file, or API credentials are missing."""


class ModelGenerationError(AIVCOModelError):
    """The model failed during generation or network request."""


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
        path_obj = Path(__file__).resolve().parent / ".env"

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


class LLMController:
    def __init__(
        self,
        llama_path: str = LLAMA_PATH,
        model_path: str = MODEL_PATH,
        env_path: str = ".env",
    ):
        self.llama_path = llama_path
        self.model_path = model_path
        self.env_path = env_path
        load_env_file(self.env_path)

    def is_available(self) -> bool:
        return os.path.exists(self.llama_path) and os.path.exists(self.model_path)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Runs one llama-cli generation and returns the generated text.
        The subprocess is started as a finite, non-interactive prompt
        so it cannot wait forever for extra user input.
        """
        if not self.is_available():
            raise ModelUnavailableError("Local LLM executable or model file is missing.")

        try:
            completed = subprocess.run(
                [
                    self.llama_path,
                    "-m",
                    self.model_path,
                    "--system-prompt",
                    system_prompt,
                    "-p",
                    user_prompt,
                    "-n",
                    str(LLM_MAX_TOKENS),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise ModelGenerationError(
                f"Model generation timed out after {LLM_TIMEOUT_SECONDS} seconds."
            ) from exc
        except OSError as exc:
            raise ModelGenerationError(
                f"Unable to execute local model binary: {exc}"
            ) from exc

        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "no output").strip()
            raise ModelGenerationError(
                f"Model generation failed with exit code {completed.returncode}: {details}"
            )

        return completed.stdout.strip()

    def _generate_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "openai/gpt-oss-120b",
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
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
    ) -> str:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ModelUnavailableError("GEMINI_API_KEY is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AIVCO/2.0",
        }
        body = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
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

    def route_and_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        task_type: str = "general",
    ) -> dict:
        """
        Routes the prompt to the appropriate model tier with automatic fallback.
        Returns:
            {"response": str, "provider_used": str, "fallback_triggered": bool}
        """
        # Rule 1: Tier 1 Strict Local (no fallback)
        if task_type in ("private", "memory", "offline"):
            response = self.generate(system_prompt, user_prompt)
            return {
                "response": response,
                "provider_used": "local_llama",
                "fallback_triggered": False,
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
                response = provider_fn(system_prompt, user_prompt)
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
