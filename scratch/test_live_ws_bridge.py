import asyncio
import json
import urllib.request
import websockets

async def test_full_bridge():
    uri = "ws://127.0.0.1:8000/ws/bridge"
    print("Connecting to WebSocket:", uri)
    async with websockets.connect(uri) as ws:
        print("Connected!")
        # 1. Send registration handshake
        handshake = {
            "type": "register",
            "client": "chrome_extension",
            "version": "1.0.0",
            "tabs": {
                "claude": True,
                "chatgpt": False,
                "gemini": False,
                "grok": False
            }
        }
        await ws.send(json.dumps(handshake))
        print("Sent registration handshake with Claude active tab.")

        # 2. Concurrently make HTTP chat request and listen for dispatch
        async def send_chat():
            await asyncio.sleep(0.5)
            print("Sending POST /api/chat...")
            req = urllib.request.Request(
                "http://127.0.0.1:8000/api/chat",
                data=json.dumps({"prompt": "write a python function to compute factorial", "target_model": "claude"}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, urllib.request.urlopen, req)
            body = resp.read().decode("utf-8")
            print("Chat Response received:\n", body)

        async def listen_and_respond():
            print("Listening on WebSocket for dispatch...")
            msg = await ws.recv()
            data = json.loads(msg)
            print("WebSocket received dispatch:", data)
            assert data["action"] == "dispatch"
            assert data["provider"] == "claude"
            assert "factorial" in data["prompt"]

            # Reply back with simulated AI response
            reply = {
                "request_id": data["request_id"],
                "status": "done",
                "markdown": "Here is the python function:\n```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n```"
            }
            await ws.send(json.dumps(reply))
            print("Replied back to server with Markdown answer!")

        await asyncio.gather(send_chat(), listen_and_respond())

asyncio.run(test_full_bridge())

