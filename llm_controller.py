"""
AIVCO LLM Controller

Thin wrapper around the llama.cpp CLI. This is the ONLY place that
uses subprocess in the whole app -- because it's the only place that
actually needs to: running the model binary. It knows nothing about
memory or commands, so upgrading from Qwen3-0.6B to a 7B/9B model
later just means changing LLAMA_PATH/MODEL_PATH (or passing new
paths into LLMController) -- the rest of AIVCO is untouched.
"""

import subprocess

LLAMA_PATH = r"C:\Users\koira\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-cli.exe"

MODEL_PATH = r"C:\Users\koira\.cache\huggingface\hub\models--ggml-org--Qwen3-0.6B-GGUF\snapshots\b5f37287796e5be0ea3dab2e7430873fb3f73e49\Qwen3-0.6B-Q4_0.gguf"


class LLMController:
    def __init__(self, llama_path: str = LLAMA_PATH, model_path: str = MODEL_PATH):
        self.llama_path = llama_path
        self.model_path = model_path

    def generate(self, system_prompt: str, user_prompt: str) -> None:
        """
        Runs one llama-cli generation. This is exactly the same
        subprocess call your working version used for real questions
        -- unchanged. Only command input is now guaranteed to never
        reach this method.
        """
        subprocess.run(
            [
                self.llama_path,
                "-m",
                self.model_path,
                "--system-prompt",
                system_prompt,
                "-p",
                user_prompt,
            ]
        )