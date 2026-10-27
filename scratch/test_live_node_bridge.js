const WebSocket = globalThis.WebSocket;

async function run() {
  console.log("Connecting from Node to ws://127.0.0.1:8000/ws/bridge...");
  const ws = new WebSocket("ws://127.0.0.1:8000/ws/bridge");

  ws.onopen = async () => {
    console.log("Node WebSocket client connected!");
    // Send register handshake
    ws.send(JSON.stringify({
      type: "register",
      client: "node_test_client",
      version: "1.0.0",
      tabs: { claude: true, chatgpt: true, gemini: false, grok: false }
    }));
    console.log("Sent handshake!");

    // Check /api/browser/extension_status
    setTimeout(async () => {
      const statusRes = await fetch("http://127.0.0.1:8000/api/browser/extension_status");
      const status = await statusRes.json();
      console.log("Status API reports:", status);

      // Trigger chat request
      console.log("Triggering POST /api/chat for Claude...");
      const chatPromise = fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: "Write factorial in Python", target_model: "claude" })
      }).then(r => r.json());

      const chatResult = await chatPromise;
      console.log("Chat Response successfully returned:\n", JSON.stringify(chatResult, null, 2));

      ws.close();
      process.exit(0);
    }, 500);
  };

  ws.onmessage = (event) => {
    console.log("Received WebSocket dispatch from server:", event.data);
    const data = JSON.parse(event.data);
    if (data.action === "dispatch") {
      // Simulate response
      setTimeout(() => {
        ws.send(JSON.stringify({
          request_id: data.request_id,
          status: "done",
          markdown: "```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n```"
        }));
        console.log("Simulated response sent over WebSocket!");
      }, 200);
    }
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
  };
}

run();

