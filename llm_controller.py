"""
AIVCO LLM Controller

Thin wrapper around the llama.cpp CLI. This is the ONLY place that
uses subprocess in the whole app -- because it's the only place that
actually needs to: running the model binary. It knows nothing about
memory or commands, so upgrading from Qwen3-0.6B to a 7B/9B model
later just means changing LLAMA_PATH/MODEL_PATH (or passing new
paths into LLMController) -- the rest of AIVCO is untouched.
"""

import os
import subprocess

LLAMA_PATH = r"C:\Users\koira\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-cli.exe"

MODEL_PATH = r"C:\Users\koira\.cache\huggingface\hub\models--ggml-org--Qwen3-0.6B-GGUF\snapshots\b5f37287796e5be0ea3dab2e7430873fb3f73e49\Qwen3-0.6B-Q4_0.gguf"
LLM_TIMEOUT_SECONDS = 180
LLM_MAX_TOKENS = 32


class AIVCOModelError(RuntimeError):
    """Base exception for expected local model failures."""


class ModelUnavailableError(AIVCOModelError):
    """The local executable or model file is missing."""


class ModelGenerationError(AIVCOModelError):
    """The local model failed during generation."""


class LLMController:
    def __init__(self, llama_path: str = LLAMA_PATH, model_path: str = MODEL_PATH):
        self.llama_path = llama_path
        self.model_path = model_path

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