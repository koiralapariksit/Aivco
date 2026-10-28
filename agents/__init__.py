"""
AIVCO Agents Package
Contains autonomous browser agents, web automation, and live search engines.
"""

from agents import browser_agent
from agents.browser_agent import (
    browser_agent as default_browser_agent,
    ExtensionBridgeRouter,
    search_web,
    universal_web_agent,
    extract_clean_markdown_from_html,
    clean_search_query,
    get_vault_status_summary,
    set_active_claude_slot,
)

__all__ = [
    "browser_agent",
    "default_browser_agent",
    "ExtensionBridgeRouter",
    "search_web",
    "universal_web_agent",
    "extract_clean_markdown_from_html",
    "clean_search_query",
    "get_vault_status_summary",
    "set_active_claude_slot",
]
