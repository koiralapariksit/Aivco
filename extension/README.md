# AIVCO Web AI Bridge — Chrome Extension (Manifest V3)

This unpacked Chrome extension bridges AIVCO directly to your authenticated AI web sessions (Claude, ChatGPT, Gemini, Grok) inside your standard Google Chrome browser.

## Why This Architecture?
- **Zero Bot Detection**: Eliminates Cloudflare Turnstile blocks, Google/OpenAI security challenges, and headless fingerprinting. Prompts execute inside your real user session with authentic cookies.
- **Zero Profile Locking**: Windows `SingletonLock` errors are completely avoided because AIVCO doesn't launch conflicting browser binaries.
- **Real-Time WebSocket**: Communicates bidirectionally over `ws://127.0.0.1:8000/ws/bridge` with automatic reconnection.

## Quick Installation (3 Steps)
1. In Google Chrome, navigate to: `chrome://extensions`
2. Enable **Developer mode** using the toggle switch in the top-right corner.
3. Click **Load unpacked** in the top-left and select this `extension/` directory (`E:\Aivco\Ai-assist\Aivco\extension`).

## Usage
- Keep your usual tabs open for [Claude](https://claude.ai), [ChatGPT](https://chatgpt.com), [Gemini](https://gemini.google.com/app), or [Grok](https://grok.com).
- Click the AIVCO extension icon in your Chrome toolbar to verify the bridge status (🟢 **Connected**) and see which AI tabs are detected.
- In the AIVCO Web HUD (`http://127.0.0.1:8000`), the top indicator will turn green:
  `🟢 EXTENSION: CONNECTED (Claude, ChatGPT, ...)`
- Any prompt sent to Claude, ChatGPT, Gemini Web, or Grok will be executed live and the formatted response will stream back into AIVCO.

