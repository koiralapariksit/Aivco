"""
Unit tests for AIVCO Arbiter Output Cleaner Fixes (Tool 7: 7.1 to 7.5).
Verifies:
- Safe script cleaner preserves JavaScript idioms and strips real TypeScript.
- Allow-list prose normalizer preserves technical identifiers and fixes known smashes/dates.
- Unfenced HTML wrapping and truncation detection with ledger protection.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel.arbiter import (
    sanitize_script_content,
    normalize_text_and_whitespace,
    sanitize_and_validate_code_output,
    SessionExecutionLedger,
    TS_SIGNALS,
    KNOWN_SMASHES,
)


def test_script_cleaner_preserves_js_idioms():
    """
    7.1 & 7.2: Verify that standard JavaScript idioms, template strings, imports with aliases,
    and comparison operators are completely preserved and never corrupted by TS stripping.
    """
    idioms_js = (
        'import { createApp as make } from "vue";\n'
        "const a = 'Mark as watched';\n"
        "const b = `Continue as ${name}`;\n"
        "if (count < max && total > (limit)) {\n"
        '    console.log("Welcome!");\n'
        "}"
    )
    cleaned = sanitize_script_content(idioms_js)
    assert cleaned == idioms_js
    assert 'import { createApp as make } from "vue";' in cleaned
    assert "Mark as watched" in cleaned
    assert "`Continue as ${name}`" in cleaned
    assert "count < max && total > (limit)" in cleaned


def test_script_cleaner_strips_real_typescript():
    """
    7.1 & 7.2: Verify genuine TypeScript patterns (type annotations, assertions, interfaces, generics)
    are cleanly stripped when TS patterns are detected.
    """
    ts_code = (
        "interface User {\n"
        "    id: number;\n"
        "    name: string;\n"
        "}\n"
        "const x: number = 5;\n"
        "(el as HTMLElement).click();\n"
        "const btn = container!.querySelector<HTMLButtonElement>('.action');\n"
        "function handle(id: number, active: boolean = false): void {\n"
        "    console.log(id, active);\n"
        "}"
    )
    cleaned = sanitize_script_content(ts_code)
    assert "interface User" not in cleaned
    assert "const x = 5;" in cleaned
    assert "el.click();" in cleaned
    assert "as HTMLElement" not in cleaned
    assert "container.querySelector" in cleaned
    assert "<HTMLButtonElement>" not in cleaned
    assert "function handle(id, active = false)" in cleaned


def test_text_cleaner_preserves_identifiers():
    """
    7.3: Verify that technical identifiers, camelCase terms, and dotted paths
    are preserved without aggressive generic camelCase splitting, while known entity
    smashes and date spacing are normalized.
    """
    identifiers = [
        "JavaScript",
        "GitHub",
        "YouTube",
        "FastAPI",
        "ChatGPT",
        "OpenAI",
        "macOS",
        "System.out.println",
    ]
    for ident in identifiers:
        sample = f"We are integrating {ident} into our production pipeline."
        cleaned = normalize_text_and_whitespace(sample)
        assert ident in cleaned, f"Expected {ident} to be preserved, got: {cleaned}"

    # Verify KNOWN_SMASHES and date normalization
    smashed = (
        "FCBarcelona defeated AtléticoMadrid on 20May2026. "
        "The next tournament is June2024. "
        "The analysis completed.Sir, here are the metrics."
    )
    cleaned_smashed = normalize_text_and_whitespace(smashed)
    assert "FC Barcelona" in cleaned_smashed
    assert "Atlético Madrid" in cleaned_smashed
    assert "20 May 2026" in cleaned_smashed
    assert "June 2024" in cleaned_smashed
    assert "completed. Sir, here are the metrics." in cleaned_smashed

    # Verify normalize_prose gate on sanitize_and_validate_code_output
    raw_code_output = "```html\n<div>JavaScript & GitHub</div>\n```"
    san_code = sanitize_and_validate_code_output(raw_code_output, normalize_prose=False)
    assert "JavaScript & GitHub" in san_code


def test_unfenced_html_and_truncation_detection(tmp_path):
    """
    7.4 & 7.5: Verify unfenced HTML is auto-wrapped in ```html fences,
    truncated outputs are detected and cleanly closed with [AIVCO-TRUNCATED],
    and ledger record_turn flags has_html_artifact = False on truncation.
    """
    # 1. Unfenced HTML wrapping (7.4)
    unfenced = "<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Hello</h1></body></html>"
    wrapped = sanitize_and_validate_code_output(unfenced)
    assert wrapped.startswith("```html\n<!DOCTYPE html>")
    assert wrapped.endswith("</html>\n```")

    # 2. Truncation Detection & Clean Closure (7.5)
    truncated_raw = (
        "```html\n"
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<body>\n"
        "<h1>Unfinished App</h1>\n"
        "<script>\n"
        "function initialize() {\n"
        "    const count = 10;\n"
    )
    closed = sanitize_and_validate_code_output(truncated_raw)
    assert "[AIVCO-TRUNCATED] Output looks cut off. Say 'continue' to finish it." in closed
    assert "</html>\n```" in closed

    # 3. Ledger Protection: has_html_artifact must be False on truncation
    ledger = SessionExecutionLedger(projects_dir=tmp_path)
    rec = ledger.record_turn(
        prompt="build game",
        target_tier="claude",
        actual_provider="groq",
        response_markdown=closed,
        mode="casual",
    )
    assert rec.has_html_artifact is False
    assert rec.html_code == ""
    assert "[AIVCO-TRUNCATED]" in rec.response

    # Verify complete HTML correctly retains has_html_artifact = True
    complete_app = (
        "```html\n"
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<body><h1>Complete App</h1></body>\n"
        "</html>\n"
        "```"
    )
    rec_complete = ledger.record_turn(
        prompt="build full game",
        target_tier="claude",
        actual_provider="groq",
        response_markdown=complete_app,
        mode="casual",
    )
    assert rec_complete.has_html_artifact is True
    assert "Complete App" in rec_complete.html_code
