"""
AIVCO Kernel Package
Provides sub-second cognitive arbitration, command routing, and zero-LLM resolution.
"""

from kernel.arbiter import (
    KernelCognitiveArbiter,
    build_executive_envelope,
    compress_memory_context,
    log_arbiter_telemetry,
    flush_telemetry,
    set_telemetry_filepath,
    synthesize_elite_prompt,
    extract_fuzzy_provider_directive,
    provider_health,
    sanitize_and_validate_code_output,
    normalize_text_and_whitespace,
    session_ledger,
    ATTRIBUTION_PATTERN,
    IDENTITY_PATTERN,
    META_CONVERSATION_EXCLUDE,
    LIVE_SEARCH_PATTERN,
    FRONTEND_PATTERN,
)
from kernel import arbiter
from kernel import cognitive_kernel

__all__ = [
    "KernelCognitiveArbiter",
    "build_executive_envelope",
    "compress_memory_context",
    "log_arbiter_telemetry",
    "flush_telemetry",
    "set_telemetry_filepath",
    "synthesize_elite_prompt",
    "extract_fuzzy_provider_directive",
    "provider_health",
    "sanitize_and_validate_code_output",
    "normalize_text_and_whitespace",
    "session_ledger",
    "ATTRIBUTION_PATTERN",
    "IDENTITY_PATTERN",
    "META_CONVERSATION_EXCLUDE",
    "LIVE_SEARCH_PATTERN",
    "FRONTEND_PATTERN",
    "arbiter",
    "cognitive_kernel",
]
