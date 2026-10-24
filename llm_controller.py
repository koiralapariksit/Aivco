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
    Reads line-by-line, ignores empty lines and comments (#),
    splits on first '=' and sets os.environ using setdefault.
    """
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
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
    except OSError:
        pass


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
        model: str = "llama-3.3-70b-versatile",
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
        model: str = "gemini-2.5-flash",
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

        last_error = None
        for idx, (provider_name, provider_fn) in enumerate(chain):
            try:
                response = provider_fn(system_prompt, user_prompt)
                return {
                    "response": response,
                    "provider_used": provider_name,
                    "fallback_triggered": idx > 0,
                }
            except AIVCOModelError as exc:
                last_error = exc
                continue

        if last_error:
            raise last_error
        raise ModelUnavailableError("No model provider available.")
