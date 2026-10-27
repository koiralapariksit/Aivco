/**
 * AIVCO Web AI Bridge — Background Service Worker (Manifest V3)
 * Manages WebSocket bridge connection to AIVCO backend (ws://127.0.0.1:8000/ws/bridge)
 * Dispatches prompts to authenticated AI web tabs (Claude, ChatGPT, Gemini, Grok)
 * and relays Markdown responses back to AIVCO.
 */

const BACKEND_WS_URL = "ws://127.0.0.1:8000/ws/bridge";
const BACKEND_HTTP_URL = "http://127.0.0.1:8000/api/bridge/result";

let socket = null;
let isConnected = false;
let reconnectTimer = null;
let heartbeatTimer = null;
const activeDispatches = new Map(); // requestId -> { tabId, keepAliveInterval }

const PROVIDER_CONFIG = {
  claude: {
    url: "https://claude.ai/new",
    matchPatterns: ["*://claude.ai/*", "*://*.claude.ai/*"],
    domains: ["claude.ai"]
  },
  chatgpt: {
    url: "https://chatgpt.com/",
    matchPatterns: ["*://chatgpt.com/*", "*://*.chatgpt.com/*"],
    domains: ["chatgpt.com", "openai.com"]
  },
  gemini: {
    url: "https://gemini.google.com/app",
    matchPatterns: ["*://gemini.google.com/*"],
    domains: ["gemini.google.com"]
  },
  gemini_web: {
    url: "https://gemini.google.com/app",
    matchPatterns: ["*://gemini.google.com/*"],
    domains: ["gemini.google.com"]
  },
  grok: {
    url: "https://grok.com/",
    matchPatterns: ["*://grok.com/*", "*://*.grok.com/*", "*://x.ai/*", "*://*.x.ai/*"],
    domains: ["grok.com", "x.ai"]
  }
};

const PROVIDER_DOMAINS = {
  claude: ["claude.ai"],
  chatgpt: ["chatgpt.com", "openai.com"],
  gemini: ["gemini.google.com"],
  grok: ["grok.com", "x.ai"]
};

const PROVIDER_URLS = {
  claude: "https://claude.ai/new",
  chatgpt: "https://chatgpt.com/",
  gemini: "https://gemini.google.com/app",
  gemini_web: "https://gemini.google.com/app",
  grok: "https://grok.com/"
};

// ── WebSocket Connection Management ──────────────────────────────────────────

function connectWebSocket() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  try {
    socket = new WebSocket(BACKEND_WS_URL);

    socket.onopen = async () => {
      isConnected = true;
      console.log("[AIVCO Bridge] Connected to AIVCO backend at", BACKEND_WS_URL);
      
      // Send registration handshake
      const tabs = await detectOpenAITabs();
      socket.send(JSON.stringify({
        type: "register",
        client: "chrome_extension",
        version: "1.0.0",
        tabs: tabs
      }));

      startHeartbeat();
    };

    socket.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.action === "pong") {
          return;
        }
        handleServerMessage(data);
      } catch (err) {
        console.error("[AIVCO Bridge] Failed to parse message from server:", err);
      }
    };

    socket.onclose = () => {
      isConnected = false;
      stopHeartbeat();
      console.warn("[AIVCO Bridge] Disconnected from AIVCO backend. Retrying in 3s...");
      scheduleReconnect();
    };

    socket.onerror = (err) => {
      console.warn("[AIVCO Bridge] WebSocket error:", err);
      try { socket.close(); } catch (_) {}
    };
  } catch (err) {
    console.warn("[AIVCO Bridge] Failed to create WebSocket:", err);
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    connectWebSocket();
  }, 3000);
}

function startHeartbeat() {
  stopHeartbeat();
  heartbeatTimer = setInterval(async () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      try {
        socket.send(JSON.stringify({ action: "ping" }));
      } catch (_) {}
      const tabs = await detectOpenAITabs();
      socket.send(JSON.stringify({
        type: "tabs_status",
        tabs: tabs
      }));
    }
  }, 5000);
}

function stopHeartbeat() {
  if (heartbeatTimer) clearInterval(heartbeatTimer);
  heartbeatTimer = null;
}

// ── Tab Discovery & Auto-Spawning ───────────────────────────────────────────

async function detectOpenAITabs() {
  const tabs = await chrome.tabs.query({});
  const status = {
    claude: false,
    chatgpt: false,
    gemini: false,
    grok: false
  };

  for (const tab of tabs) {
    if (!tab.url) continue;
    const url = tab.url.toLowerCase();
    for (const [provider, config] of Object.entries(PROVIDER_CONFIG)) {
      if (provider === "gemini_web") continue;
      if (config.domains && config.domains.some(domain => url.includes(domain))) {
        status[provider] = true;
      }
    }
  }

  return status;
}

async function getOrCreateTabForProvider(provider) {
  const norm = (provider === "gemini_web" || provider === "gemini-web") ? "gemini" : provider.toLowerCase();
  const config = PROVIDER_CONFIG[norm] || PROVIDER_CONFIG[provider];
  if (!config) throw new Error(`Unknown provider: ${provider}`);

  // 1. Check for an existing open tab
  let tabs = [];
  try {
    tabs = await chrome.tabs.query({ url: config.matchPatterns });
  } catch (err) {
    console.warn(`[AIVCO Bridge] tabs.query by pattern failed, using fallback scan:`, err);
  }

  // Fallback domain scan if matchPatterns returned empty
  if (!tabs || tabs.length === 0) {
    const allTabs = await chrome.tabs.query({});
    tabs = allTabs.filter(t => {
      if (!t.url) return false;
      const u = t.url.toLowerCase();
      const doms = config.domains || [norm];
      return doms.some(d => u.includes(d));
    });
  }

  if (tabs && tabs.length > 0) {
    const activeTab = tabs[0];
    // Keep tab running silently in background (Never steal focus)
    return activeTab;
  }

  // 2. No tab found -> Automatically spawn it silently in the background
  console.log(`[AIVCO Bridge] No active tab for ${provider}. Creating in background: ${config.url}`);
  const newTab = await chrome.tabs.create({ url: config.url, active: false });

  // 3. Wait for the new tab to finish loading
  await new Promise((resolve) => {
    function listener(tabId, info) {
      if (tabId === newTab.id && info.status === 'complete') {
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }
    }
    chrome.tabs.onUpdated.addListener(listener);
    // 15-second safety timeout
    setTimeout(() => {
      chrome.tabs.onUpdated.removeListener(listener);
      resolve();
    }, 15000);
  });

  // Give React/ProseMirror 2.5 seconds to mount its state
  await new Promise((r) => setTimeout(r, 2500));
  return newTab;
}

// ── Prompt Dispatch Execution ────────────────────────────────────────────────

async function handleServerMessage(data) {
  const action = data.action;
  const requestId = data.request_id;
  const provider = data.provider;
  const prompt = data.prompt;

  if (action === "dispatch") {
    console.log(`[AIVCO Bridge] Received dispatch for ${provider} (Req ID: ${requestId})`);
    try {
      const targetTab = await getOrCreateTabForProvider(provider);
      // Forward to content script in the identified tab
      executePromptInTab(targetTab.id, provider, prompt, requestId);
    } catch (err) {
      sendResponseToServer({
        request_id: requestId,
        status: "error",
        error: `Failed to acquire tab for ${provider}: ${err.message}`
      });
    }
  }
}

function deliverBridgeResult(requestId, provider, markdown, error = null) {
  if (!activeDispatches.has(requestId) && !requestId) return;

  if (activeDispatches.has(requestId)) {
    const session = activeDispatches.get(requestId);
    if (session.keepAliveInterval) clearInterval(session.keepAliveInterval);
    if (session.monitorInterval) clearInterval(session.monitorInterval);
    activeDispatches.delete(requestId);
  }

  // 1. Deliver via WebSocket
  sendResponseToServer({
    action: "dispatch_result",
    request_id: requestId,
    provider: provider,
    status: error ? "error" : "done",
    markdown: markdown || "",
    error: error || null
  });

  // 2. Direct HTTP POST callback to /api/bridge/result
  try {
    fetch(BACKEND_HTTP_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request_id: requestId,
        status: error ? "error" : "done",
        markdown: markdown || "",
        error: error || null,
        provider: provider
      })
    }).then(res => {
      console.log(`[AIVCO Bridge] HTTP callback status for ${requestId}:`, res.status);
    }).catch(err => {
      console.warn(`[AIVCO Bridge] HTTP callback error:`, err);
    });
  } catch (e) {
    console.warn(`[AIVCO Bridge] HTTP fetch dispatch failed:`, e);
  }
}

// ── Standalone DOM Inspector for executeScript ───────────────────────────────

function captureTabBaseline() {
  try {
    const assistantSelectors = [
      '.font-claude-message',
      '[data-message-author-role="assistant"]',
      'model-response-text',
      '.model-response-text',
      'message-content.model-response-text',
      '[data-message-role="assistant"]',
      '.agent-turn [data-message-author-role="assistant"]',
      'div[data-is-streaming="false"]',
      '[data-test-render-count]',
      '.prose'
    ];
    const userSelectors = '[data-testid="user-message"], .font-user-message, [data-message-author-role="user"], .user-message, fieldset, form';
    let assistantEls = [];
    for (const sel of assistantSelectors) {
      const els = Array.from(document.querySelectorAll(sel)).filter(el => !el.matches(userSelectors) && !el.closest(userSelectors));
      if (els.length > 0) {
        assistantEls = els;
        break;
      }
    }
    const lastEl = assistantEls.length > 0 ? assistantEls[assistantEls.length - 1] : null;
    const lastMsgText = lastEl ? (lastEl.innerText || "").trim() : "";
    const preEls = lastEl ? lastEl.querySelectorAll('pre, code') : document.querySelectorAll('pre, code');
    let lastCode = "";
    if (preEls.length > 0) {
      const lastPre = preEls[preEls.length - 1];
      lastCode = (lastPre.textContent || "").trim().slice(0, 120);
    }
    return {
      baselineMsgCount: assistantEls.length,
      baselineLastMsgText: lastMsgText,
      baselineCodeSnippet: lastCode
    };
  } catch (_) {
    return { baselineMsgCount: 0, baselineLastMsgText: "", baselineCodeSnippet: "" };
  }
}

function inspectPageState(promptText, baseline) {
  try {
    // 0. Detect Rate-Limit Banners immediately
    const rateLimitRegex = /(reached your .*limit|message limit|try again after|upgrade to claude pro|out of free messages|too many requests|usage limit)/i;
    const bodyText = document.body ? (document.body.innerText || '') : '';
    if (rateLimitRegex.test(bodyText)) {
      const banner = Array.from(document.querySelectorAll('div, p, span, h2, h3, [role="alert"]')).find(el => {
        const t = (el.innerText || '').trim();
        return t.length < 350 && rateLimitRegex.test(t);
      });
      if (banner) {
        return {
          isRateLimited: true,
          error: "RATE_LIMIT_REACHED",
          totalLength: 0,
          codeBlocks: [],
          artifactHtml: '',
          proseText: ''
        };
      }
    }

    // 1. Check if streaming is active
    const stopSelectors = [
      'button[aria-label*="Stop" i]',
      'button[data-testid="stop-button"]',
      '[data-is-streaming="true"]',
      'button.stop-button',
      'button[aria-label*="Cancel" i]'
    ];
    let isStreaming = false;
    for (const sel of stopSelectors) {
      const els = document.querySelectorAll(sel);
      for (const el of els) {
        const rect = el.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          isStreaming = true;
          break;
        }
      }
      if (isStreaming) break;
    }

    // 2. Identify Assistant Message Nodes
    const assistantSelectors = [
      '.font-claude-message',
      '[data-message-author-role="assistant"]',
      'model-response-text',
      '.model-response-text',
      'message-content.model-response-text',
      '[data-message-role="assistant"]',
      '.agent-turn [data-message-author-role="assistant"]',
      'div[data-is-streaming="false"]',
      '[data-test-render-count]',
      '.prose',
      'main article'
    ];
    const userSelectors = '[data-testid="user-message"], .font-user-message, [data-message-author-role="user"], .user-message, fieldset, form';
    let assistantEls = [];
    for (const sel of assistantSelectors) {
      const els = Array.from(document.querySelectorAll(sel)).filter(el =>
        !el.matches(userSelectors) && !el.closest(userSelectors)
      );
      if (els.length > 0) {
        assistantEls = els;
        break;
      }
    }

    const currentMsgCount = assistantEls.length;
    const latestAssistantEl = assistantEls.length > 0 ? assistantEls[assistantEls.length - 1] : null;
    let currentProseRaw = latestAssistantEl ? (latestAssistantEl.innerText || "").trim() : "";

    // Baseline Stale-DOM Check:
    let hasNewTurn = false;
    if (baseline) {
      if (currentMsgCount > baseline.baselineMsgCount) {
        hasNewTurn = true;
      } else if (currentProseRaw.length > 20 && currentProseRaw !== baseline.baselineLastMsgText && !currentProseRaw.startsWith(baseline.baselineLastMsgText)) {
        hasNewTurn = true;
      }
    } else {
      hasNewTurn = currentProseRaw.length > 20;
    }

    // If no new turn has started and not currently streaming, do not scrape old stale response
    if (!hasNewTurn && !isStreaming) {
      return {
        isStreaming: false,
        codeBlocks: [],
        artifactHtml: '',
        proseText: '',
        totalLength: 0,
        hasNewResponse: false
      };
    }

    // 3. Extract code blocks RESTRICTED strictly to latest assistant container!
    const codeBlocks = [];
    const codeScope = latestAssistantEl || document;
    const preEls = codeScope.querySelectorAll('pre, .cm-content');
    for (const pre of preEls) {
      if (pre.closest(userSelectors)) continue;
      const codeEl = pre.querySelector('code') || pre;
      const text = (codeEl.textContent || '').trim();

      // Filter out Cloudflare challenge scripts / parameters
      if (text.includes('__CF$cv$params') || text.includes('/cdn-cgi/challenge-platform/') || text.includes('cf-turnstile')) {
        continue;
      }
      // Filter out baseline code snippet if turn hasn't incremented
      if (baseline && currentMsgCount <= baseline.baselineMsgCount && text.slice(0, 120) === baseline.baselineCodeSnippet) {
        continue;
      }

      if (text.length > 15) {
        let lang = 'html';
        if (text.startsWith('<!DOCTYPE') || text.startsWith('<html') || text.includes('</html') || text.includes('<style') || text.includes('<body')) {
          lang = 'html';
        } else if (text.startsWith('{') || text.startsWith('[')) {
          lang = 'json';
        } else if (text.includes('import ') || text.includes('export default') || text.includes('function ')) {
          lang = 'javascript';
        } else if (text.includes('def ') || text.includes('class ') || text.includes('def __init__')) {
          lang = 'python';
        }
        codeBlocks.push({ lang, code: text });
      }
    }

    // 4. Extract any Artifact iframe HTML
    let artifactHtml = '';
    const iframes = (latestAssistantEl || document).querySelectorAll('iframe');
    for (const iframe of iframes) {
      try {
        let html = iframe.getAttribute('srcdoc') || '';
        if (!html && iframe.contentDocument && iframe.contentDocument.documentElement) {
          html = iframe.contentDocument.documentElement.outerHTML || '';
        }
        if (html && (html.includes('__CF$cv$params') || html.includes('challenge-platform') || html.includes('cf-turnstile'))) {
          continue;
        }
        if (html && html.trim().length > 100) {
          artifactHtml = html.trim();
          break;
        }
      } catch (_) {}
    }

    // 5. Extract assistant prose text
    let proseText = '';
    if (latestAssistantEl) {
      const clone = latestAssistantEl.cloneNode(true);
      clone.querySelectorAll('pre, iframe, ' + userSelectors).forEach(e => e.remove());
      let txt = (clone.innerText || '').trim();
      txt = txt.replace(/\[AIVCO EXECUTIVE BRIEF\][\s\S]*?\[\/AIVCO EXECUTIVE BRIEF\]/gi, '').trim();
      txt = txt.replace(/^(?:Claude responded:?\s*)?/i, '').trim();
      if (txt.length > 15) {
        proseText = txt;
      }
    }

    const totalLength = proseText.length + artifactHtml.length + codeBlocks.reduce((acc, c) => acc + c.code.length, 0);

    return {
      isStreaming,
      codeBlocks,
      artifactHtml,
      proseText,
      totalLength,
      hasNewResponse: hasNewTurn || totalLength > 20
    };
  } catch (err) {
    return {
      isStreaming: false,
      codeBlocks: [],
      artifactHtml: '',
      proseText: '',
      totalLength: 0,
      hasNewResponse: false,
      error: err.message
    };
  }
}

function assembleMarkdownFromResult(res) {
  let finalMd = "";
  if (res.proseText) {
    let cleanProse = res.proseText.replace(/^(?:Claude responded:?\s*)?/i, '').trim();
    cleanProse = cleanProse.replace(/\[AIVCO EXECUTIVE BRIEF\][\s\S]*?\[\/AIVCO EXECUTIVE BRIEF\]/gi, '').trim();
    finalMd += cleanProse;
  }
  if (res.artifactHtml) {
    finalMd += (finalMd ? "\n\n" : "") + "```html\n" + res.artifactHtml + "\n```";
  }
  for (const b of res.codeBlocks) {
    if (!res.artifactHtml || !res.artifactHtml.includes(b.code.slice(0, 50))) {
      finalMd += (finalMd ? "\n\n" : "") + "```" + b.lang + "\n" + b.code + "\n```";
    }
  }
  return finalMd.replace(/\[AIVCO EXECUTIVE BRIEF\][\s\S]*?\[\/AIVCO EXECUTIVE BRIEF\]/gi, '').trim();
}

// ── Active Supervisor Polling Loop ───────────────────────────────────────────

function monitorTabGeneration(tabId, requestId, provider, promptText, baseline) {
  let baselineLength = null;
  let lastLength = 0;
  let stableCount = 0;
  let hasIncreased = false;
  let pollCount = 0;
  const maxPolls = 75; // 150 seconds

  const monitorInterval = setInterval(async () => {
    if (!activeDispatches.has(requestId)) {
      clearInterval(monitorInterval);
      return;
    }

    pollCount++;
    if (pollCount > maxPolls) {
      console.warn(`[AIVCO Bridge] Monitor reached max polls for ${requestId}. Forcing extraction...`);
      clearInterval(monitorInterval);
      try {
        const results = await chrome.scripting.executeScript({
          target: { tabId },
          func: inspectPageState,
          args: [promptText, baseline]
        });
        if (results && results[0] && results[0].result) {
          const res = results[0].result;
          const md = assembleMarkdownFromResult(res);
          deliverBridgeResult(requestId, provider, md || "[aivco-kernel] Execution timed out waiting for AI response.");
        }
      } catch (_) {
        deliverBridgeResult(requestId, provider, "", "Execution timed out waiting for AI response.");
      }
      return;
    }

    try {
      const results = await chrome.scripting.executeScript({
        target: { tabId },
        func: inspectPageState,
        args: [promptText, baseline]
      });

      if (!results || !results[0] || !results[0].result) return;
      const res = results[0].result;

      // Rate limit check
      if (res.isRateLimited || res.error === "RATE_LIMIT_REACHED") {
        console.warn(`[AIVCO Bridge] Rate limit detected for ${requestId}! Aborting wait immediately.`);
        clearInterval(monitorInterval);
        deliverBridgeResult(requestId, provider, "", "RATE_LIMIT_REACHED");
        return;
      }

      if (baselineLength === null) {
        baselineLength = res.totalLength;
        lastLength = res.totalLength;
        console.log(`[AIVCO Bridge] Baseline page length for ${requestId}: ${baselineLength}`);
        return;
      }

      // Detect if content has started growing or if new turn mounted
      if (res.hasNewResponse && (res.totalLength > baselineLength + 20 || res.artifactHtml.length > 50 || res.codeBlocks.length > 0)) {
        hasIncreased = true;
      }

      // Check completion: Content increased AND not streaming AND stable for 2 polls (4 seconds)
      if (hasIncreased && !res.isStreaming) {
        if (Math.abs(res.totalLength - lastLength) < 10) {
          stableCount++;
          console.log(`[AIVCO Bridge] Content stable for poll ${stableCount}/2 (Len: ${res.totalLength})`);
          if (stableCount >= 2) {
            console.log(`[AIVCO Bridge] Active monitor detected generation completion for ${requestId}!`);
            clearInterval(monitorInterval);
            const finalMd = assembleMarkdownFromResult(res);
            deliverBridgeResult(requestId, provider, finalMd);
            return;
          }
        } else {
          stableCount = 0;
          lastLength = res.totalLength;
        }
      } else {
        lastLength = res.totalLength;
        stableCount = 0;
      }
    } catch (err) {
      console.log(`[AIVCO Bridge] Monitor poll notice for tab ${tabId}:`, err.message);
    }
  }, 2000);

  return monitorInterval;
}

async function executePromptInTab(tabId, provider, prompt, requestId, retryCount = 0) {
  // Capture pre-flight DOM baseline snapshot to prevent re-scraping stale messages
  let baseline = { baselineMsgCount: 0, baselineLastMsgText: "", baselineCodeSnippet: "" };
  try {
    const snapResults = await chrome.scripting.executeScript({
      target: { tabId },
      func: captureTabBaseline
    });
    if (snapResults && snapResults[0] && snapResults[0].result) {
      baseline = snapResults[0].result;
      console.log(`[AIVCO Bridge] Captured DOM baseline for tab ${tabId}:`, baseline);
    }
  } catch (err) {
    console.warn("[AIVCO Bridge] Could not capture pre-flight baseline:", err);
  }

  // 1. Keep background service worker alive during generation
  const keepAliveInterval = setInterval(() => {
    try {
      chrome.runtime.getPlatformInfo(() => {});
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ action: "progress", request_id: requestId }));
      }
    } catch (_) {}
  }, 2000);

  // 2. Start active tab supervisor polling loop
  const monitorInterval = monitorTabGeneration(tabId, requestId, provider, prompt, baseline);

  activeDispatches.set(requestId, { tabId, keepAliveInterval, monitorInterval });

  // 175-second safety timer to clear if unhandled
  setTimeout(() => {
    if (activeDispatches.has(requestId)) {
      const s = activeDispatches.get(requestId);
      if (s.keepAliveInterval) clearInterval(s.keepAliveInterval);
      if (s.monitorInterval) clearInterval(s.monitorInterval);
      activeDispatches.delete(requestId);
    }
  }, 175000);

  chrome.tabs.sendMessage(
    tabId,
    { action: "execute_prompt", provider, prompt, request_id: requestId, baseline },
    async (ack) => {
      if (chrome.runtime.lastError) {
        console.warn(`[AIVCO Bridge] Initial sendMessage error on tab ${tabId}:`, chrome.runtime.lastError.message);
        // If content script was not ready yet, inject dynamically and retry once
        if (retryCount === 0) {
          console.log(`[AIVCO Bridge] Content script not detected on tab ${tabId}. Injecting dynamically...`);
          try {
            await chrome.scripting.executeScript({
              target: { tabId: tabId },
              files: ["content.js"]
            });
            clearInterval(keepAliveInterval);
            clearInterval(monitorInterval);
            activeDispatches.delete(requestId);
            setTimeout(() => {
              executePromptInTab(tabId, provider, prompt, requestId, 1);
            }, 600);
            return;
          } catch (injectErr) {
            console.error("[AIVCO Bridge] Failed to inject content script:", injectErr);
          }
        }

        deliverBridgeResult(requestId, provider, "", `Tab connection error on ${provider}: ${chrome.runtime.lastError.message}. Please refresh the tab.`);
        return;
      }

      console.log(`[AIVCO Bridge] Prompt dispatched & acknowledged by tab (Req ID: ${requestId}):`, ack);
    }
  );
}

function sendResponseToServer(payload) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  } else {
    console.error("[AIVCO Bridge] Cannot send response, WebSocket not open:", payload);
  }
}

// ── Tab State Change Listeners ───────────────────────────────────────────────

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete") {
    const tabs = await detectOpenAITabs();
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "tabs_status", tabs }));
    }
  }
});

chrome.tabs.onRemoved.addListener(async () => {
  const tabs = await detectOpenAITabs();
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "tabs_status", tabs }));
  }
});

// ── Extension & Content Script Communication ─────────────────────────────────

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  // Push-event return from content script upon generation completion
  if (request.action === "dispatch_result") {
    const reqId = request.request_id;
    console.log(`[AIVCO Bridge] Received dispatch_result from content script for ${reqId} (status: ${request.status})`);
    deliverBridgeResult(reqId, request.provider, request.markdown, request.error);
    sendResponse({ forwarded: true });
    return false;
  }

  if (request.action === "keepalive") {
    try {
      chrome.runtime.getPlatformInfo(() => {});
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ action: "progress", request_id: request.request_id }));
      }
    } catch (_) {}
    sendResponse({ ok: true });
    return false;
  }

  if (request.action === "get_status") {
    detectOpenAITabs().then((tabs) => {
      sendResponse({
        connected: isConnected,
        tabs: tabs,
        wsUrl: BACKEND_WS_URL
      });
    });
    return true; // Keep channel open for async response
  }
});

// Start connection on service worker init
connectWebSocket();

