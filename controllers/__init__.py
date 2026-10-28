"""
AIVCO Controllers Package
Handles LLM execution, model routing, and deterministic command processing.
"""

from controllers import llm_controller
from controllers import command_controller
from controllers.llm_controller import (
    LLMController,
    ModelUnavailableError,
    ModelGenerationError,
    AIVCOModelError,
    load_env_file,
    ANTI_HALLUCINATION_MANDATE,
    DEFAULT_SYSTEM_PROMPT,
)
from controllers.command_controller import (
    CommandController,
    HELP_TEXT,
    EXIT_SENTINEL,
    CLEAR_SENTINEL,
)

__all__ = [
    "llm_controller",
    "command_controller",
    "LLMController",
    "ModelUnavailableError",
    "ModelGenerationError",
    "AIVCOModelError",
    "load_env_file",
    "ANTI_HALLUCINATION_MANDATE",
    "DEFAULT_SYSTEM_PROMPT",
    "CommandController",
    "HELP_TEXT",
    "EXIT_SENTINEL",
    "CLEAR_SENTINEL",
]
