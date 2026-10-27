"""
AIVCO Brain 4 — Native Chrome Extension Bridge Router (Manifest V3) & Live Web Grounding
Replaces brittle Playwright/CDP browser scraping with an unpacked Chrome Extension
communicating over local WebSockets (ws://127.0.0.1:8000/ws/bridge).
Eliminates Cloudflare Turnstile bot detection and profile lock contention.
Includes hands-free OS browser auto-launching, persistent MV3 keep-alive heartbeats,
and 3-stage live web search grounding (DuckDuckGo + Wikipedia).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import html
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger("aivco.extension_bridge")

# ── HTML to Clean Markdown Preserver ──────────────────────────────────────────

def extract_clean_markdown_from_html(html_str: str) -> str:
    """
    Converts HTML response into clean Markdown while strictly preserving
    <pre><code> blocks as fenced ``` code blocks.
    """
    if not html_str:
        return ""

    blocks = {}
    counter = 0

    def code_repl(m):
        nonlocal counter
        lang = m.group(1) or ""
        raw_code = m.group(2)
        code_text = html.unescape(raw_code).strip()
        token = f"___AIVCO_CODE_BLOCK_{counter}___"
        blocks[token] = f"\n```{lang}\n{code_text}\n```\n"
        counter += 1
        return token

    # Extract <pre><code ...>...</code></pre>
    res = re.sub(
        r'<pre[^>]*><code(?: class="[^"]*language-(\w+)[^"]*")?[^>]*>(.*?)</code></pre>',
        code_repl,
        html_str,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Extract <pre>...</pre> without <code>
    def pre_only_repl(m):
        nonlocal counter
        raw_code = m.group(1)
        code_text = html.unescape(raw_code).strip()
        token = f"___AIVCO_CODE_BLOCK_{counter}___"
        blocks[token] = f"\n```\n{code_text}\n```\n"
        counter += 1
        return token

    res = re.sub(
        r'<pre[^>]*>(?!<code)(.*?)</pre>',
        pre_only_repl,
        res,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Convert block elements to newlines
    res = re.sub(r'</?(?:p|div|section|article|header|footer)[^>]*>', '\n', res, flags=re.IGNORECASE)
    res = re.sub(r'<br\s*/?>', '\n', res, flags=re.IGNORECASE)
    res = re.sub(r'<hr\s*/?>', '\n---\n', res, flags=re.IGNORECASE)

    # Convert headings
    res = re.sub(r'<h1[^>]*>(.*?)</h1>', r'\n# \1\n', res, flags=re.IGNORECASE | re.DOTALL)
    res = re.sub(r'<h2[^>]*>(.*?)</h2>', r'\n## \1\n', res, flags=re.IGNORECASE | re.DOTALL)
    res = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n### \1\n', res, flags=re.IGNORECASE | re.DOTALL)
    res = re.sub(r'<h4[^>]*>(.*?)</h4>', r'\n#### \1\n', res, flags=re.IGNORECASE | re.DOTALL)

    # Lists
    res = re.sub(r'<li[^>]*>(.*?)</li>', r'\n- \1', res, flags=re.IGNORECASE | re.DOTALL)
    res = re.sub(r'</?[ou]l[^>]*>', '\n', res, flags=re.IGNORECASE)

    # Inline formatting
    res = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', res, flags=re.IGNORECASE | re.DOTALL)
    res = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', res, flags=re.IGNORECASE | re.DOTALL)
    res = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', res, flags=re.IGNORECASE | re.DOTALL)
    res = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', res, flags=re.IGNORECASE | re.DOTALL)
    res = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', res, flags=re.IGNORECASE | re.DOTALL)

    # Remove all remaining HTML tags
    res = re.sub(r'<[^>]+>', '', res)

    # Decode HTML entities
    res = html.unescape(res)

    # Restore preserved code blocks
    for token, block in blocks.items():
        res = res.replace(token, block)

    # Normalize excessive newlines
    res = re.sub(r'\n{3,}', '\n\n', res).strip()
    return res


# ── Web AI Provider Metadata & Target URLs ────────────────────────────────────

PROVIDER_URLS: Dict[str, str] = {
    "claude": "https://claude.ai/new",
    "chatgpt": "https://chatgpt.com/",
    "gemini": "https://gemini.google.com/app",
    "gemini_web": "https://gemini.google.com/app",
    "grok": "https://grok.com/",
}

TARGET_URLS: Dict[str, str] = dict(PROVIDER_URLS)

PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
    "claude": {
        "url": "https://claude.ai/new",
        "matchPatterns": ["*://claude.ai/*", "*://*.claude.ai/*"],
        "domains": ["claude.ai"],
    },
    "chatgpt": {
        "url": "https://chatgpt.com/",
        "matchPatterns": ["*://chatgpt.com/*", "*://*.chatgpt.com/*"],
        "domains": ["chatgpt.com", "openai.com"],
    },
    "gemini": {
        "url": "https://gemini.google.com/app",
        "matchPatterns": ["*://gemini.google.com/*"],
        "domains": ["gemini.google.com"],
    },
    "gemini_web": {
        "url": "https://gemini.google.com/app",
        "matchPatterns": ["*://gemini.google.com/*"],
        "domains": ["gemini.google.com"],
    },
    "grok": {
        "url": "https://grok.com/",
        "matchPatterns": ["*://grok.com/*", "*://*.grok.com/*", "*://x.ai/*", "*://*.x.ai/*"],
        "domains": ["grok.com", "x.ai"],
    },
}

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "claude": {"name": "Claude", "url": "https://claude.ai", "domains": ["claude.ai"]},
    "chatgpt": {"name": "ChatGPT", "url": "https://chatgpt.com", "domains": ["chatgpt.com", "openai.com"]},
    "gemini": {"name": "Gemini", "url": "https://gemini.google.com/app", "domains": ["gemini.google.com"]},
    "gemini_web": {"name": "Gemini", "url": "https://gemini.google.com/app", "domains": ["gemini.google.com"]},
    "grok": {"name": "Grok", "url": "https://grok.com", "domains": ["grok.com", "x.ai"]},
}


# ── Extension Bridge Router ───────────────────────────────────────────────────

class ExtensionBridgeRouter:
    """
    Coordinates real-time communication between AIVCO and the Chrome Extension
    over WebSocket (ws://127.0.0.1:8000/ws/bridge).
    Supports hands-free OS browser launch when no client is currently connected.
    """

    def __init__(self):
        self._clients: Set[Any] = set()
        self._lock = threading.Lock()
        self._connected_tabs: Dict[str, Any] = {
            "claude": False,
            "chatgpt": False,
            "gemini": False,
            "grok": False,
        }
        self._pending_requests: Dict[str, concurrent.futures.Future] = {}
        self._server_loop: Optional[asyncio.AbstractEventLoop] = None

    def register_client(self, websocket: Any) -> None:
        """Registers a connected Chrome extension WebSocket client."""
        with self._lock:
            self._clients.add(websocket)
            try:
                self._server_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        logger.info("[bridge] Chrome Extension connected successfully. Total clients: %d", len(self._clients))

    def unregister_client(self, websocket: Any) -> None:
        """Removes a disconnected Chrome extension WebSocket client."""
        with self._lock:
            self._clients.discard(websocket)
            if not self._clients:
                for k in self._connected_tabs:
                    self._connected_tabs[k] = False
        logger.info("[bridge] Chrome Extension disconnected. Total clients: %d", len(self._clients))

    def is_connected(self) -> bool:
        """Checks if at least one Chrome extension client is active."""
        with self._lock:
            return len(self._clients) > 0

    def get_connected_tabs(self) -> Dict[str, Any]:
        """Returns the current tab availability for each web AI provider."""
        with self._lock:
            return dict(self._connected_tabs)

    def update_tabs_status(self, tabs_status: Dict[str, Any]) -> None:
        """Updates available provider tabs reported by the extension."""
        with self._lock:
            for k, v in tabs_status.items():
                norm = "gemini" if k in ("gemini_web", "gemini-web") else k.lower()
                self._connected_tabs[norm] = bool(v)

    def resolve_request(
        self,
        request_id: Optional[str] = None,
        markdown: str = "",
        error: Optional[str] = None,
    ) -> bool:
        """
        Resolves a pending request Future whether the completion payload arrives
        via WebSocket (/ws/bridge) or direct HTTP POST (/api/bridge/result).
        If request_id is 'latest', None, or omitted, resolves the most recent pending request.
        """
        with self._lock:
            fut: Optional[concurrent.futures.Future] = None
            target_id: Optional[str] = None

            if request_id and request_id in self._pending_requests:
                target_id = request_id
                fut = self._pending_requests.pop(request_id)
            elif (not request_id or request_id in ("latest", "auto")) and self._pending_requests:
                target_id = list(self._pending_requests.keys())[-1]
                fut = self._pending_requests.pop(target_id)
            elif len(self._pending_requests) == 1:
                target_id = next(iter(self._pending_requests.keys()))
                fut = self._pending_requests.pop(target_id)

            if fut and not fut.done():
                if error:
                    fut.set_exception(RuntimeError(error))
                else:
                    fut.set_result(markdown or "")
                logger.info("[bridge] Resolved pending request %s (markdown len: %d)", target_id, len(markdown or ""))
                return True

            return False

    def handle_incoming_message(self, data: Dict[str, Any]) -> None:
        """
        Processes incoming WebSocket payloads from the Chrome extension:
        - Handshakes / registrations
        - Tab status updates
        - Task responses: { "request_id": id, "status": "done"|"error", "markdown": "...", "error": "..." }
        """
        msg_type = data.get("type") or data.get("action")
        req_id = data.get("request_id")

        if msg_type in ("register", "handshake"):
            logger.info("[bridge] Received handshake from extension: %s", data)
            if "tabs" in data and isinstance(data["tabs"], dict):
                self.update_tabs_status(data["tabs"])
            return

        if msg_type == "tabs_status" or "tabs" in data:
            tabs = data.get("tabs", {})
            if isinstance(tabs, dict):
                self.update_tabs_status(tabs)
            return

        if msg_type == "progress":
            return

        if msg_type in ("dispatch_result", "result") or (req_id and req_id in self._pending_requests):
            status = data.get("status", "done")
            err_msg = data.get("error") if status == "error" else None
            markdown = data.get("markdown", "")
            self.resolve_request(req_id, markdown=markdown, error=err_msg)
            return

    def _launch_browser_for_provider(self, provider: str) -> None:
        """Launches the default browser to the provider URL in background without stealing focus."""
        is_testing = "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))
        if is_testing:
            return

        target_url = PROVIDER_URLS.get(provider, "https://claude.ai/new")
        logger.info("[bridge] Auto-launching browser for %s at %s...", provider, target_url)
        try:
            # Use autoraise=False to avoid stealing operator focus
            webbrowser.open(target_url, new=2, autoraise=False)
        except Exception as e:
            logger.warning("[bridge] Failed to auto-launch browser: %s", e)

    async def _wait_for_connection_async(self, timeout: float = 15.0) -> bool:
        """Polls asynchronously for up to timeout seconds waiting for extension to connect."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_connected():
                return True
            await asyncio.sleep(0.5)
        return self.is_connected()

    def _wait_for_connection_sync(self, timeout: float = 15.0) -> bool:
        """Polls synchronously for up to timeout seconds waiting for extension to connect."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_connected():
                return True
            time.sleep(0.5)
        return self.is_connected()

    async def dispatch_prompt_async(
        self,
        provider: str,
        prompt: str,
        timeout: float = 180.0,
        auto_launch_wait: float = 15.0,
    ) -> str:
        """
        Dispatches a prompt asynchronously across the WebSocket bridge
        and awaits the completed Markdown response (default 180s timeout).
        If not connected, automatically opens the browser and waits up to auto_launch_wait.
        """
        norm_provider = "gemini" if provider in ("gemini_web", "gemini-web") else provider.lower()

        # Hands-free auto-launch if disconnected
        if not self.is_connected():
            is_testing = "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))
            if auto_launch_wait > 0 and not is_testing:
                self._launch_browser_for_provider(norm_provider)
                await self._wait_for_connection_async(timeout=auto_launch_wait)

        if not self.is_connected():
            return (
                f"[aivco-kernel] Chrome Extension bridge is not connected. "
                f"Please load the unpacked extension in chrome://extensions and keep {norm_provider} open in a tab."
            )

        request_id = f"req-{uuid.uuid4().hex[:10]}"
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._pending_requests[request_id] = fut

        payload = {
            "action": "dispatch",
            "provider": norm_provider,
            "prompt": prompt,
            "request_id": request_id,
        }
        msg_str = json.dumps(payload)

        # Broadcast to all connected extension clients
        sent = False
        with self._lock:
            clients = list(self._clients)

        for client in clients:
            try:
                if hasattr(client, "send_text"):
                    await client.send_text(msg_str)
                    sent = True
                elif hasattr(client, "send"):
                    await client.send(msg_str)
                    sent = True
            except Exception as e:
                logger.warning("[bridge] Failed to send to client: %s", e)
                self.unregister_client(client)

        if not sent:
            self._pending_requests.pop(request_id, None)
            return (
                f"[aivco-kernel] Chrome Extension bridge client disconnected during dispatch. "
                f"Please verify Chrome is open and reload the extension."
            )

        loop = asyncio.get_running_loop()
        try:
            res = await asyncio.wait_for(
                loop.run_in_executor(None, fut.result, timeout),
                timeout=timeout + 2.0,
            )
            return res or f"[aivco-kernel] {norm_provider.capitalize()} returned an empty response."
        except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
            self._pending_requests.pop(request_id, None)
            return (
                f"[aivco-kernel] Timeout waiting for response from Chrome Extension ({int(timeout)}s). "
                f"Ensure the {norm_provider} tab is active and not showing a CAPTCHA."
            )
        except Exception as exc:
            self._pending_requests.pop(request_id, None)
            return f"[aivco-kernel] Chrome Extension error: {str(exc)}"

    def query(
        self,
        provider: str,
        prompt: str,
        timeout: float = 180.0,
        auto_launch_wait: float = 15.0,
    ) -> str:
        """
        Synchronous query method compatible with synchronous callers (e.g. llm_controller.py).
        Safely bridges to the async dispatch engine with auto-launch capability.
        """
        norm_provider = "gemini" if provider in ("gemini_web", "gemini-web") else provider.lower()

        # Hands-free auto-launch if disconnected
        if not self.is_connected():
            is_testing = "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))
            if auto_launch_wait > 0 and not is_testing:
                self._launch_browser_for_provider(norm_provider)
                self._wait_for_connection_sync(timeout=auto_launch_wait)

        if not self.is_connected():
            return (
                f"[aivco-kernel] Chrome Extension bridge is not connected. "
                f"Please load the unpacked extension in chrome://extensions and keep {norm_provider} open in a tab."
            )

        request_id = f"req-{uuid.uuid4().hex[:10]}"
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._pending_requests[request_id] = fut

        payload = {
            "action": "dispatch",
            "provider": norm_provider,
            "prompt": prompt,
            "request_id": request_id,
        }
        msg_str = json.dumps(payload)

        with self._lock:
            clients = list(self._clients)
            target_loop = self._server_loop

        sent = False
        for client in clients:
            try:
                if hasattr(client, "send_text"):
                    if target_loop and target_loop.is_running():
                        asyncio.run_coroutine_threadsafe(client.send_text(msg_str), target_loop)
                        sent = True
                    else:
                        asyncio.run(client.send_text(msg_str))
                        sent = True
            except Exception as e:
                logger.warning("[bridge] Failed to dispatch query to client: %s", e)

        if not sent:
            if not self.is_connected():
                self._pending_requests.pop(request_id, None)
                return (
                    f"[aivco-kernel] Chrome Extension bridge is not connected. "
                    f"Please load the unpacked extension in chrome://extensions and keep {norm_provider} open in a tab."
                )

        try:
            res = fut.result(timeout=timeout)
            return res or f"[aivco-kernel] {norm_provider.capitalize()} returned an empty response."
        except concurrent.futures.TimeoutError:
            self._pending_requests.pop(request_id, None)
            return (
                f"[aivco-kernel] Timeout waiting for response from Chrome Extension ({int(timeout)}s). "
                f"Ensure the {norm_provider} tab is active and not showing a CAPTCHA."
            )
        except Exception as exc:
            self._pending_requests.pop(request_id, None)
            return f"[aivco-kernel] Chrome Extension error: {str(exc)}"

    def status(self) -> Dict[str, Any]:
        """Status summary of the Extension Bridge."""
        connected = self.is_connected()
        tabs = self.get_connected_tabs()
        return {
            "running": connected,
            "connected": connected,
            "client_count": len(self._clients),
            "tabs": tabs,
            "mode": "chrome_extension_manifest_v3",
        }

    def launch_cdp(self) -> Dict[str, Any]:
        """Backward compatibility stub."""
        return {
            "ok": True,
            "status": "deprecated",
            "message": "CDP mode has been migrated to AIVCO Native Chrome Extension Bridge (Manifest V3).",
        }


# Global singleton instance
extension_bridge_router = ExtensionBridgeRouter()
universal_web_agent = extension_bridge_router
browser_agent = extension_bridge_router


# ── Backward Compatibility Vault Functions ────────────────────────────────────

def get_vault_status_summary() -> Dict[str, Any]:
    """Provides vault summary reporting extension bridge state."""
    connected = extension_bridge_router.is_connected()
    tabs = extension_bridge_router.get_connected_tabs()
    return {
        "ok": True,
        "extension_connected": connected,
        "mode": "extension_v3",
        "providers": {
            "claude": {"status": "ready" if tabs.get("claude") else ("connected" if connected else "offline")},
            "chatgpt": {"status": "ready" if tabs.get("chatgpt") else ("connected" if connected else "offline")},
            "gemini": {"status": "ready" if tabs.get("gemini") else ("connected" if connected else "offline")},
            "grok": {"status": "ready" if tabs.get("grok") else ("connected" if connected else "offline")},
        },
        "claude_vault": {
            "active_slot": 1,
            "total_slots": 20,
            "status": "ready" if tabs.get("claude") else ("connected" if connected else "offline"),
            "slots": [
                {
                    "slot": 1,
                    "status": "ready" if tabs.get("claude") else ("connected" if connected else "offline"),
                    "is_active": True,
                    "is_authenticated": connected,
                }
            ],
        },
    }

def set_active_claude_slot(slot: int) -> Dict[str, Any]:
    return {"ok": True, "active_slot": slot, "message": "Using Chrome Extension Bridge"}

def get_active_claude_slot() -> int:
    return 1

def is_cdp_running() -> bool:
    return extension_bridge_router.is_connected()

def launch_native_browser_sessions(profile_dir: Optional[str] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "message": "Playwright scraping retired. Please use AIVCO Chrome Extension.",
    }


# ── 3-Stage Live Web Search Grounding ─────────────────────────────────────────

def clean_search_query(query: str) -> str:
    """Extracts the core search keywords from natural language prompts."""
    q = query.strip()
    q = re.sub(
        r'^(?:search\s+(?:the\s+web\s+for|for|google\s+for)?|look\s+up|google|find|who\s+is|what\s+is|tell\s+me\s+about)\s+',
        '',
        q,
        flags=re.IGNORECASE,
    ).strip()
    return q or query.strip()


def search_web(query: str, max_results: int = 6) -> str:
    """
    3-Stage Grounding:
    1. DuckDuckGo Instant Answer API (JSON)
    2. DuckDuckGo HTML Lite scraping fallback
    3. Wikipedia API enrichment
    """
    clean_q = clean_search_query(query)
    results: List[Dict[str, str]] = []

    # Stage 1: DuckDuckGo Instant Answer API
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_q)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "AIVCO-Argon/2.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            abstract = data.get("AbstractText", "")
            if abstract:
                src_url = data.get("AbstractURL", "")
                results.append({"title": f"Instant Answer: {clean_q}", "snippet": abstract, "url": src_url})

            for related in data.get("RelatedTopics", [])[:3]:
                if isinstance(related, dict) and "Text" in related:
                    results.append({"title": clean_q, "snippet": related["Text"], "url": related.get("FirstURL", "")})
    except Exception:
        pass

    # Stage 2: DuckDuckGo HTML Lite Fallback
    if len(results) < 2:
        try:
            lite_url = "https://lite.duckduckgo.com/lite/"
            post_data = urllib.parse.urlencode({"q": clean_q}).encode("utf-8")
            req = urllib.request.Request(
                lite_url,
                data=post_data,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw_html = resp.read().decode("utf-8", errors="ignore")
                matches = re.findall(
                    r'<td class="result-snippet">\s*(.*?)\s*</td>',
                    raw_html,
                    re.DOTALL | re.IGNORECASE,
                )
                link_matches = re.findall(
                    r'<a class="result-link" href="([^"]+)">\s*(.*?)\s*</a>',
                    raw_html,
                    re.DOTALL | re.IGNORECASE,
                )
                for i, snip in enumerate(matches[:max_results]):
                    clean_snip = re.sub(r'<[^>]+>', '', snip).strip()
                    title = "Web Result"
                    href = ""
                    if i < len(link_matches):
                        href = link_matches[i][0]
                        title = re.sub(r'<[^>]+>', '', link_matches[i][1]).strip()
                    if clean_snip:
                        results.append({"title": title, "snippet": clean_snip, "url": href})
        except Exception:
            pass

    # Stage 3: Wikipedia API enrichment
    if len(results) < 3:
        try:
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean_q.replace(' ', '_'))}"
            req = urllib.request.Request(wiki_url, headers={"User-Agent": "AIVCO-Argon/2.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                wiki_data = json.loads(resp.read().decode("utf-8"))
                extract = wiki_data.get("extract")
                if extract:
                    page_url = wiki_data.get("content_urls", {}).get("desktop", {}).get("page", "")
                    results.append({"title": f"Wikipedia: {wiki_data.get('title', clean_q)}", "snippet": extract, "url": page_url})
        except Exception:
            pass

    if not results:
        return f"[Web Search] No live search results found for query: '{clean_q}'."

    formatted = []
    for i, r in enumerate(results[:max_results], 1):
        url_part = f" ({r['url']})" if r.get("url") else ""
        formatted.append(f"[{i}] {r['title']}{url_part}:\n{r['snippet']}")

    return "\n\n".join(formatted)
