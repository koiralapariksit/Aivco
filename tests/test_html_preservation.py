import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from agents.browser_agent import ExtensionBridgeRouter, extract_clean_markdown_from_html

@pytest.mark.anyio
async def test_websocket_markdown_preserves_html_tags():
    router = ExtensionBridgeRouter()
    mock_ws = AsyncMock()
    router.register_client(mock_ws)

    sample_html = """<!DOCTYPE html>
<html>
<head>
    <style>body{background:red;}</style>
</head>
<body>
    <h1>Test</h1>
</body>
</html>"""

    sample_markdown = f"Here is the website you requested:\n```html\n{sample_html}\n```\nEnjoy!"

    async def fake_extension():
        await asyncio.sleep(0.02)
        # Find the request_id from the mock call
        call_args = mock_ws.send_text.call_args[0][0]
        import json
        payload = json.loads(call_args)
        req_id = payload["request_id"]
        router.handle_incoming_message({
            "request_id": req_id,
            "status": "done",
            "markdown": sample_markdown
        })

    task = asyncio.create_task(fake_extension())
    result = await router.dispatch_prompt_async("claude", "generate site", timeout=5.0)
    await task

    # Assert that 100% of the raw HTML tags are preserved intact
    assert "<!DOCTYPE html>" in result
    assert "<html>" in result
    assert "<head>" in result
    assert "<style>body{background:red;}</style>" in result
    assert "<body>" in result
    assert "<h1>Test</h1>" in result
    assert "</body>" in result
    assert "</html>" in result
    print("All HTML tags 100% preserved in WebSocket dispatch response!")

if __name__ == "__main__":
    asyncio.run(test_websocket_markdown_preserves_html_tags())

