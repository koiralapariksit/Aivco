"""
Unit and integration tests for AIVCO Native Chrome Extension Bridge (Manifest V3).
Tests ExtensionBridgeRouter, WebSocket message handling, tab tracking, and prompt dispatch.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from agents.browser_agent import (
    ExtensionBridgeRouter,
    extract_clean_markdown_from_html,
    clean_search_query,
)


@pytest.fixture
def bridge_router():
    """Returns an isolated instance of ExtensionBridgeRouter."""
    return ExtensionBridgeRouter()


def test_initial_state(bridge_router):
    assert bridge_router.is_connected() is False
    tabs = bridge_router.get_connected_tabs()
    assert tabs["claude"] is False
    assert tabs["chatgpt"] is False
    assert tabs["gemini"] is False
    assert tabs["grok"] is False


def test_client_registration_and_tab_tracking(bridge_router):
    mock_ws = MagicMock()
    bridge_router.register_client(mock_ws)
    assert bridge_router.is_connected() is True

    # Receive registration message from extension
    bridge_router.handle_incoming_message({
        "type": "register",
        "client": "chrome_extension",
        "version": "1.0.0",
        "tabs": {
            "claude": True,
            "chatgpt": True,
            "gemini": False,
            "grok": False
        }
    })

    tabs = bridge_router.get_connected_tabs()
    assert tabs["claude"] is True
    assert tabs["chatgpt"] is True
    assert tabs["gemini"] is False

    # Unregister
    bridge_router.unregister_client(mock_ws)
    assert bridge_router.is_connected() is False
    assert bridge_router.get_connected_tabs()["claude"] is False


def test_query_when_disconnected(bridge_router):
    res = bridge_router.query("claude", "hello")
    assert "[aivco-kernel] Chrome Extension bridge is not connected" in res
    assert "chrome://extensions" in res


@pytest.mark.anyio
async def test_dispatch_prompt_lifecycle(bridge_router):
    mock_ws = AsyncMock()
    bridge_router.register_client(mock_ws)

    async def fake_extension_responder():
        await asyncio.sleep(0.05)
        # Verify the mock received the dispatch payload
        assert mock_ws.send_text.called
        call_args = mock_ws.send_text.call_args[0][0]
        data = json.loads(call_args)
        assert data["action"] == "dispatch"
        assert data["provider"] == "claude"
        assert data["prompt"] == "write a react component"
        req_id = data["request_id"]

        # Simulate extension answering back with markdown
        bridge_router.handle_incoming_message({
            "request_id": req_id,
            "status": "done",
            "markdown": "```jsx\nexport default function App() { return <h1>Hello</h1>; }\n```"
        })

    # Run dispatch and mock responder concurrently
    responder_task = asyncio.create_task(fake_extension_responder())
    result = await bridge_router.dispatch_prompt_async("claude", "write a react component", timeout=5.0)
    await responder_task

    assert "export default function App()" in result
    assert "```jsx" in result


def test_markdown_extraction():
    html_sample = """
    <div>
        <h2>Title Here</h2>
        <p>This is a paragraph with <b>bold</b> text and <code>inline code</code>.</p>
        <pre><code class="language-python">def add(a, b):
    return a + b</code></pre>
    </div>
    """
    md = extract_clean_markdown_from_html(html_sample)
    assert "## Title Here" in md
    assert "**bold**" in md
    assert "`inline code`" in md
    assert "```python\ndef add(a, b):\n    return a + b\n```" in md


def test_clean_search_query():
    assert clean_search_query("search for python tutorials") == "python tutorials"
    assert clean_search_query("who is Lionel Messi") == "Lionel Messi"
    assert clean_search_query("what is quantum computing") == "quantum computing"


def test_progress_message_handling(bridge_router):
    # progress message should be safely ignored and not raise any error
    bridge_router.handle_incoming_message({
        "action": "progress",
        "request_id": "req-12345"
    })
    assert True


@pytest.mark.anyio
async def test_auto_launch_wait_when_disconnected(bridge_router):
    # In pytest environment, is_testing is True so it doesn't wait 15s or open real browser
    res = await bridge_router.dispatch_prompt_async("claude", "test prompt", auto_launch_wait=15.0)
    assert "[aivco-kernel] Chrome Extension bridge is not connected" in res
    assert "claude" in res


@pytest.mark.anyio
async def test_dispatch_result_push_event(bridge_router):
    mock_ws = AsyncMock()
    bridge_router.register_client(mock_ws)

    async def fake_push_responder():
        await asyncio.sleep(0.05)
        call_args = mock_ws.send_text.call_args[0][0]
        data = json.loads(call_args)
        req_id = data["request_id"]

        # Push-event return from extension via dispatch_result
        bridge_router.handle_incoming_message({
            "action": "dispatch_result",
            "request_id": req_id,
            "provider": "claude",
            "status": "done",
            "markdown": "Here is the code:\n```html\n<!DOCTYPE html><html><body><h1>Push Event Works</h1></body></html>\n```"
        })

    responder_task = asyncio.create_task(fake_push_responder())
    result = await bridge_router.dispatch_prompt_async("claude", "build a tactical room", timeout=5.0)
    await responder_task

    assert "Push Event Works" in result
    assert "<!DOCTYPE html>" in result
    assert "```html" in result


@pytest.mark.anyio
async def test_receive_bridge_result_http_endpoint():
    import time
    from server.app import receive_bridge_result, universal_web_agent
    mock_ws = AsyncMock()
    universal_web_agent.register_client(mock_ws)

    dispatch_task = asyncio.create_task(
        universal_web_agent.dispatch_prompt_async("claude", "test http bridge", timeout=5.0)
    )
    await asyncio.sleep(0.02)

    call_args = mock_ws.send_text.call_args[0][0]
    data = json.loads(call_args)
    req_id = data["request_id"]

    t0 = time.time()
    endpoint_res = await receive_bridge_result({
        "request_id": req_id,
        "markdown": "```html\n<h1>Direct HTTP Delivery</h1>\n```"
    })
    elapsed = time.time() - t0

    result = await dispatch_task

    assert endpoint_res["status"] == "ok"
    assert endpoint_res["resolved"] is True
    assert elapsed < 0.05  # Resolves immediately in <50ms
    assert "Direct HTTP Delivery" in result
    assert "```html" in result


def test_resolve_request_latest_fallback(bridge_router):
    # Test resolve by 'latest' or when only 1 pending
    import concurrent.futures
    fut = concurrent.futures.Future()
    bridge_router._pending_requests["req-alpha"] = fut

    resolved = bridge_router.resolve_request("latest", markdown="fallback latest content")
    assert resolved is True
    assert fut.result() == "fallback latest content"
    assert "req-alpha" not in bridge_router._pending_requests


def test_cloudflare_challenge_script_filtering():
    raw_html = (
        '<div>'
        '<p>Valid content here</p>'
        '<script>window.__CF$cv$params={r:"8df1",t:"MTcyODc0"};</script>'
        '<pre><code>window.__CF$cv$params={r:"8df1"};</code></pre>'
        '<pre><code class="language-html"><!DOCTYPE html><html><body><h1>Safe App</h1></body></html></code></pre>'
        '</div>'
    )
    md = extract_clean_markdown_from_html(raw_html)
    assert "__CF$cv$params" not in md
    assert "Safe App" in md
    assert "<!DOCTYPE html>" in md


def test_zero_focus_steal_when_connected(bridge_router, monkeypatch):
    import webbrowser
    opened_urls = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *args, **kwargs: opened_urls.append(url))

    # When connected, _launch_browser_for_provider should NEVER call webbrowser.open
    mock_ws = MagicMock()
    bridge_router.register_client(mock_ws)
    assert bridge_router.is_connected() is True

    bridge_router._launch_browser_for_provider("claude")
    assert len(opened_urls) == 0




