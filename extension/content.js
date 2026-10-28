/**
 * AIVCO Web AI Bridge — Content Script (Manifest V3)
 * Runs directly inside web AI pages (Claude, ChatGPT, Gemini, Grok).
 * Enters prompts into the rich input DOM, triggers submission,
 * monitors generation to completion, extracts full artifact code, and returns clean Markdown.
 */

if (window.__AIVCO_CONTENT_SCRIPT_LOADED__) {
  console.log("[AIVCO Bridge] Content script already initialized in this tab. Skipping duplicate registration.");
} else {
  window.__AIVCO_CONTENT_SCRIPT_LOADED__ = true;

  (function () {
    console.log("[AIVCO Bridge] Content script loaded for:", window.location.hostname);

    // ── Provider Identification ──────────────────────────────────────────────────

    function detectProvider() {
      const host = window.location.hostname.toLowerCase();
      if (host.includes("claude.ai")) return "claude";
      if (host.includes("chatgpt.com") || host.includes("openai.com")) return "chatgpt";
      if (host.includes("gemini.google.com")) return "gemini";
      if (host.includes("grok.com") || host.includes("x.ai")) return "grok";
      return "unknown";
    }

    // ── Selector Configurations ──────────────────────────────────────────────────

    const CONFIGS = {
      claude: {
        inputSelectors: [
          "div[contenteditable='true'].ProseMirror",
          "div[contenteditable='true']",
          "fieldset div[contenteditable='true']",
          "textarea"
        ],
        sendSelectors: [
          'button[aria-label="Send message"]',
          'button[aria-label="Send Message"]',
          'button[aria-label*="Send message" i]',
          'button[aria-label*="Send" i]',
          'button[data-testid="send-button"]'
        ],
        stopSelectors: [
          "button[aria-label*='Stop']",
          "[data-is-streaming='true']"
        ],
        responseSelectors: [
          ".font-claude-message",
          "div[data-is-streaming='false']",
          "[data-test-render-count]",
          ".prose"
        ]
      },
      chatgpt: {
        inputSelectors: [
          "#prompt-textarea",
          "div[contenteditable='true']#prompt-textarea",
          "div[contenteditable='true'][id='prompt-textarea']",
          "textarea[placeholder*='Message']",
          "textarea[data-id]",
          "div.ProseMirror[contenteditable='true']",
          "div[contenteditable='true']"
        ],
        sendSelectors: [
          "button[data-testid='send-button']",
          "button[aria-label*='Send' i]",
          "button[aria-label*='Send message' i]",
          "button:has(svg[data-icon='arrow-up'])"
        ],
        stopSelectors: [
          "button[data-testid='stop-button']",
          "button[aria-label*='Stop' i]",
          "button[aria-label*='Cancel' i]"
        ],
        responseSelectors: [
          "div[data-message-author-role='assistant']",
          "[data-message-author-role='assistant']"
        ]
      },
      gemini: {
        inputSelectors: [
          "div.ql-editor[contenteditable='true']",
          "rich-textarea div[contenteditable='true']",
          "div[role='textbox']",
          "div[contenteditable='true']"
        ],
        sendSelectors: [
          "button.send-button",
          "button[aria-label*='Send' i]",
          "button[mat-icon-button]"
        ],
        stopSelectors: [
          "button.stop-button",
          "button[aria-label*='Stop' i]",
          "button[aria-label*='Cancel' i]"
        ],
        responseSelectors: [
          "message-content.model-response-text",
          ".model-response-text",
          "model-response",
          "message-content",
          ".response-container-content",
          ".markdown"
        ]
      },
      grok: {
        inputSelectors: [
          "div.tiptap.ProseMirror[contenteditable='true']",
          "textarea",
          "div[contenteditable='true']"
        ],
        sendSelectors: [
          "button[data-testid='chat-submit']",
          "button[aria-label*='Submit' i]",
          "button[aria-label*='Send' i]",
          "button[type='submit']"
        ],
        stopSelectors: [
          "button[aria-label*='Stop' i]",
          "button[aria-label*='Cancel' i]"
        ],
        responseSelectors: [
          "[data-message-role='assistant']",
          "[data-message-author-role='assistant']",
          ".response",
          ".prose"
        ]
      }
    };

    // ── Find Visible Element & Async Wait ────────────────────────────────────────

    function findElement(selectors) {
      for (const sel of selectors) {
        try {
          const els = document.querySelectorAll(sel);
          for (const el of els) {
            const rect = el.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0 && window.getComputedStyle(el).visibility !== "hidden") {
              return el;
            }
          }
        } catch (_) {}
      }
      return null;
    }

    async function waitForEditor(selectors, timeoutMs = 12000) {
      const start = Date.now();
      while (Date.now() - start < timeoutMs) {
        const el = findElement(selectors);
        if (el) return el;
        await new Promise((r) => setTimeout(r, 400));
      }
      return null;
    }

    // ── Clean Assistant DOM Element & Remove UI Noise ───────────────────────────

    function cleanAndFormatElement(element) {
      if (!element) return "";
      const clone = element.cloneNode(true);

      // 1. Extract and preserve all literal code blocks with unique tokens BEFORE any transformations
      const codeBlocks = [];
      const preElements = clone.querySelectorAll("pre");
      preElements.forEach((pre, idx) => {
        const codeEl = pre.querySelector("code");
        const lang = codeEl
          ? (codeEl.className.match(/language-(\w+)/) || ["", ""])[1]
          : "";
        // textContent on a DOM pre/code contains literal "<!DOCTYPE html><html>" characters!
        const rawCode = (codeEl ? codeEl.textContent : pre.textContent) || "";

        // Strictly filter Cloudflare challenge scripts / parameters
        if (rawCode.includes('__CF$cv$params') || rawCode.includes('/cdn-cgi/challenge-platform/') || rawCode.includes('cf-turnstile')) {
          pre.remove();
          return;
        }

        const placeholder = `___AIVCO_CODE_BLOCK_${idx}___`;
        codeBlocks.push({ placeholder, lang, code: rawCode.trim() });

        const tokenNode = document.createTextNode(`\n\n${placeholder}\n\n`);
        if (pre.parentNode) {
          pre.parentNode.replaceChild(tokenNode, pre);
        } else {
          pre.remove();
        }
      });

      // 2. Remove thinking containers, status badges, action bars, buttons, SVGs, artifact headers
      const noiseSelectors = [
        'button',
        'svg',
        '[data-testid*="thinking"]',
        '[data-testid*="action"]',
        '[data-testid*="feedback"]',
        '.font-claude-thinking',
        '[aria-label*="Thinking" i]',
        '[aria-label*="thought" i]',
        'details',
        'summary',
        '[class*="artifact-card"] button',
        '[data-testid*="artifact"] button'
      ];

      noiseSelectors.forEach(sel => {
        try {
          clone.querySelectorAll(sel).forEach(el => el.remove());
        } catch (_) {}
      });

      // 3. Format text blocks (headings, lists, inline code)
      clone.querySelectorAll("strong, b").forEach(b => { b.textContent = ` **${b.textContent.trim()}** `; });
      clone.querySelectorAll("em, i").forEach(i => { i.textContent = ` *${i.textContent.trim()}* `; });
      clone.querySelectorAll("code").forEach(c => { c.textContent = ` \`${c.textContent.trim()}\` `; });
      clone.querySelectorAll("h1").forEach(h => { h.textContent = `\n# ${h.textContent.trim()}\n`; });
      clone.querySelectorAll("h2").forEach(h => { h.textContent = `\n## ${h.textContent.trim()}\n`; });
      clone.querySelectorAll("h3").forEach(h => { h.textContent = `\n### ${h.textContent.trim()}\n`; });
      clone.querySelectorAll("li").forEach(li => { li.textContent = `\n- ${li.textContent.trim()}`; });
      clone.querySelectorAll("p").forEach(p => { p.textContent = `\n${p.textContent.trim()}\n`; });

      let text = clone.textContent || clone.innerText || "";

      // 4. Clean up noise prefixes and artifact artifacts
      text = text.replace(/^(?:Claude responded:?\s*)?(?:Sharing files|Thinking|Thinking\.\.\.|Read a file, created a file, shared files|Thought for \d+ seconds?|Reviewing an attached task spec\.\.\.|View details)[\s\n]*/gi, '');
      text = text.replace(/(Reviewing an attached task spec\.\.\.?\s*)+/gi, '');
      text = text.replace(/(Thinking\.\.\.?\s*)+/gi, '');
      text = text.replace(/\bCopy\s*html\b/gi, '');
      text = text.replace(/\bCopy\s*code\b/gi, '');
      text = text.replace(/\b(?:just now|Copy|Retry)\b\s*$/gi, '');

      // 5. Restore the preserved code blocks verbatim
      for (const item of codeBlocks) {
        let lang = item.lang || 'html';
        if (!item.lang) {
          if (item.code.startsWith('<!DOCTYPE') || item.code.startsWith('<html') || item.code.includes('<body')) {
            lang = 'html';
          }
        }
        const codeBlockMd = `\n\`\`\`${lang}\n${item.code}\n\`\`\`\n`;
        text = text.replace(item.placeholder, codeBlockMd);
      }

      return text.replace(/\n{3,}/g, '\n\n').trim();
    }

    // ── Extract Full Claude Artifact Code ────────────────────────────────────────

    function detectCodeLanguage(code) {
      const trimmed = code.trim();
      if (trimmed.startsWith('<!DOCTYPE') || trimmed.startsWith('<html') || trimmed.includes('<body') || trimmed.includes('</html')) {
        return 'html';
      }
      if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
        return 'json';
      }
      if (trimmed.includes('import React') || trimmed.includes('export default')) {
        return 'jsx';
      }
      if (trimmed.includes('import ') || trimmed.includes('export ')) {
        return 'javascript';
      }
      if (trimmed.startsWith('def ') || trimmed.includes('class ') || trimmed.includes('def __init__')) {
        return 'python';
      }
      return 'html';
    }

    async function extractClaudeArtifactCode() {
      const artifacts = [];

      // 1. Direct Artifact Iframe Extraction (Claude's live sandboxed preview iframe)
      const iframes = Array.from(document.querySelectorAll('iframe'));
      for (const iframe of iframes) {
        try {
          let htmlContent = iframe.getAttribute('srcdoc') || '';
          if (!htmlContent && iframe.contentDocument) {
            htmlContent = iframe.contentDocument.documentElement ? iframe.contentDocument.documentElement.outerHTML : '';
          }
          if (htmlContent && htmlContent.trim().length > 100) {
            const trimmed = htmlContent.trim();
            console.log(`[AIVCO Bridge] Extracted ${trimmed.length} chars of HTML directly from Artifact iframe.`);
            artifacts.push(`\`\`\`html\n${trimmed}\n\`\`\``);
            return artifacts.join("\n\n");
          }
        } catch (_) {}
      }

      // 2. Check mounted code blocks or switch to Code view ('</>' or 'Code')
      let artifactPreBlocks = Array.from(document.querySelectorAll(
        'pre code, .cm-content, [data-testid*="artifact"] pre, [class*="artifact"] pre code'
      ));

      if (artifactPreBlocks.length === 0) {
        const headerButtons = Array.from(document.querySelectorAll('button'));
        for (const btn of headerButtons) {
          const txt = (btn.textContent || '').trim();
          const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
          const title = (btn.getAttribute('title') || '').toLowerCase();
          if (aria.includes('code') || title.includes('code') || txt === '</>' || txt === 'Code') {
            console.log("[AIVCO Bridge] Switching Claude Artifact to Code view...");
            btn.click();
            await new Promise(r => setTimeout(r, 350));
            break;
          }
        }
        artifactPreBlocks = Array.from(document.querySelectorAll(
          'pre code, .cm-content, [data-testid*="artifact"] pre, [class*="artifact"] pre code'
        ));
      }

      // Extract text from mounted code blocks using textContent (preserves 100% of HTML tags!)
      for (const block of artifactPreBlocks) {
        const codeText = block.textContent || '';
        if (codeText.trim().length > 20) {
          let lang = detectCodeLanguage(codeText);
          artifacts.push(`\`\`\`${lang}\n${codeText.trim()}\n\`\`\``);
        }
      }

      // 3. Fallback: If code is behind 'Copy' or 'Export' button, click and read clipboard with 500ms timeout
      if (artifacts.length === 0) {
        const copyBtns = Array.from(document.querySelectorAll('button')).filter(b => {
          const text = (b.textContent || '').trim().toLowerCase();
          const aria = (b.getAttribute('aria-label') || '').toLowerCase();
          return text === 'copy' || text === 'export' || aria.includes('copy code') || aria.includes('copy content');
        });

        if (copyBtns.length > 0) {
          try {
            const targetCopyBtn = copyBtns[copyBtns.length - 1];
            console.log("[AIVCO Bridge] Clicking Artifact Copy/Export button:", targetCopyBtn);
            targetCopyBtn.click();
            await new Promise(r => setTimeout(r, 200));

            const clipPromise = navigator.clipboard.readText();
            const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error('clipboard timeout')), 500));
            const clipText = await Promise.race([clipPromise, timeoutPromise]);

            if (clipText && clipText.trim().length > 20) {
              const trimmed = clipText.trim();
              let lang = detectCodeLanguage(trimmed);
              artifacts.push(`\`\`\`${lang}\n${trimmed}\n\`\`\``);
              console.log(`[AIVCO Bridge] Extracted ${trimmed.length} characters of artifact code via clipboard.`);
            }
          } catch (e) {
            console.warn("[AIVCO Bridge] Clipboard read fallback skipped:", e);
          }
        }
      }

      return artifacts.join("\n\n");
    }

    async function safeExtractClaudeArtifactCode() {
      try {
        const extractionPromise = extractClaudeArtifactCode();
        const timeoutPromise = new Promise((resolve) => setTimeout(() => resolve(""), 3000));
        return await Promise.race([extractionPromise, timeoutPromise]);
      } catch (err) {
        console.warn("[AIVCO Bridge] safeExtractClaudeArtifactCode caught error:", err);
        return "";
      }
    }

    // ── Dedicated Claude Prompt Submission ───────────────────────────────────────

    async function submitClaudePrompt(editor, promptText) {
      // 0. If Claude was accidentally put into Incognito mode, exit it or navigate to normal chat
      const incognitoBanner = Array.from(document.querySelectorAll('div, span, h1, h2')).find(
        el => el.textContent && el.textContent.trim() === "You're incognito"
      );
      if (incognitoBanner) {
        console.log("[AIVCO Bridge] Detected Incognito banner in Claude. Attempting to exit...");
        const closeIncognitoBtn = document.querySelector('button[aria-label*="Close" i], button[aria-label*="Exit" i]');
        if (closeIncognitoBtn) {
          closeIncognitoBtn.click();
          await new Promise(r => setTimeout(r, 400));
        }
      }

      // Re-query editor in case DOM refreshed
      editor = document.querySelector('div.ProseMirror[contenteditable="true"]') ||
               document.querySelector('div[contenteditable="true"]');
      if (!editor) throw new Error("Claude editor not found");

      editor.focus();

      // 1. Dual text insertion strategy for background & unfocused tabs
      let inserted = false;
      try {
        inserted = document.execCommand('selectAll', false, null) &&
                   document.execCommand('insertText', false, promptText);
      } catch (_) {}

      // Fallback: Direct DOM + ProseMirror InputEvent if tab is running in background
      if (!inserted || !editor.textContent || editor.textContent.trim().length === 0) {
        editor.innerHTML = '';
        const p = document.createElement('p');
        p.textContent = promptText;
        editor.appendChild(p);
        editor.dispatchEvent(new InputEvent('input', {
          bubbles: true,
          cancelable: true,
          inputType: 'insertText',
          data: promptText
        }));
      } else {
        editor.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: promptText }));
      }

      // Wait 350ms for React state to enable the orange submit arrow button
      await new Promise(r => setTimeout(r, 350));

      // 2. Find the EXACT Submit Button (Must be inside the prompt container and at the FAR RIGHT)
      let sendButton = document.querySelector(
        'button[aria-label="Send message"], button[aria-label="Send Message"], button[data-testid="send-button"]'
      );

      if (!sendButton) {
        const promptContainer = editor.closest('fieldset') ||
                                editor.closest('form') ||
                                editor.parentElement?.parentElement?.parentElement;
        if (promptContainer) {
          const candidateButtons = Array.from(promptContainer.querySelectorAll('button')).filter(btn => {
            if (btn.disabled || getComputedStyle(btn).display === 'none') return false;
            const label = (btn.getAttribute('aria-label') || '').toLowerCase();
            const text = (btn.textContent || '').trim().toLowerCase();
            if (label.includes('attach') || label.includes('add') || label.includes('model') ||
                label.includes('voice') || label.includes('incognito') || label.includes('dictat') ||
                text === '+' || text.includes('sonnet') || text.includes('opus') || text.includes('haiku')) {
              return false;
            }
            return true;
          });

          if (candidateButtons.length > 0) {
            candidateButtons.sort((a, b) => b.getBoundingClientRect().right - a.getBoundingClientRect().right);
            sendButton = candidateButtons[0];
          }
        }
      }

      // 3. Submit via real Pointer + Mouse + Click sequence AND Enter key fallback
      if (sendButton) {
        console.log("[AIVCO Bridge] Clicking Claude Send button:", sendButton);
        sendButton.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true }));
        sendButton.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
        sendButton.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, cancelable: true }));
        sendButton.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true }));
        sendButton.click();
      }

      // Wait 300ms and check if the editor cleared (meaning prompt was sent)
      await new Promise(r => setTimeout(r, 300));
      if (editor.textContent && editor.textContent.trim().length > 0) {
        console.log("[AIVCO Bridge] Editor still has text, dispatching Enter key on ProseMirror...");
        editor.focus();
        const enterEventInit = {
          key: 'Enter',
          code: 'Enter',
          keyCode: 13,
          which: 13,
          bubbles: true,
          cancelable: true,
          composed: true
        };
        editor.dispatchEvent(new KeyboardEvent('keydown', enterEventInit));
        editor.dispatchEvent(new KeyboardEvent('keypress', enterEventInit));
        editor.dispatchEvent(new KeyboardEvent('keyup', enterEventInit));
      }
    }

    // ── Generic Prompt Submission (ChatGPT, Gemini, Grok) ───────────────────────

    function escapeHtml(str) {
      return (str || '').replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    async function submitGenericPrompt(element, text, config) {
      element.focus();

      if (element.tagName === "TEXTAREA" || element.tagName === "INPUT") {
        element.value = text;
        element.dispatchEvent(new Event("input", { bubbles: true }));
        element.dispatchEvent(new Event("change", { bubbles: true }));
      } else {
        let inserted = false;
        try {
          document.execCommand("selectAll", false, null);
          inserted = document.execCommand("insertText", false, text);
          if (inserted) {
            element.dispatchEvent(new InputEvent("input", { bubbles: true, cancelable: true, inputType: "insertText", data: text }));
            element.dispatchEvent(new Event("change", { bubbles: true }));
          }
        } catch (_) {}
        if (!inserted || !element.textContent || element.textContent.trim().length === 0) {
          element.innerHTML = '<p>' + escapeHtml(text) + '</p>';
          element.dispatchEvent(new InputEvent("input", { bubbles: true, cancelable: true, inputType: "insertText", data: text }));
          element.dispatchEvent(new Event("change", { bubbles: true }));
        }
      }

      await new Promise(r => setTimeout(r, 300));

      const sendBtn = findElement(config.sendSelectors);
      if (sendBtn && !sendBtn.disabled) {
        sendBtn.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true }));
        sendBtn.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
        sendBtn.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, cancelable: true }));
        sendBtn.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true }));
        sendBtn.click();
      } else {
        const enterInit = {
          key: "Enter",
          code: "Enter",
          keyCode: 13,
          which: 13,
          bubbles: true,
          cancelable: true,
          composed: true
        };
        element.dispatchEvent(new KeyboardEvent("keydown", enterInit));
        element.dispatchEvent(new KeyboardEvent("keypress", enterInit));
        element.dispatchEvent(new KeyboardEvent("keyup", enterInit));
      }
    }

    function getAssistantMessages(config) {
      const userSelectors = '[data-testid="user-message"], .font-user-message, [data-message-author-role="user"], .user-message, fieldset, form';
      for (const sel of config.responseSelectors) {
        try {
          const els = Array.from(document.querySelectorAll(sel)).filter(el =>
            !el.matches(userSelectors) && !el.closest(userSelectors)
          );
          if (els.length > 0) {
            return els;
          }
        } catch (_) {}
      }
      return [];
    }

    // ── Response Observer ────────────────────────────────────────────────────────

    async function waitForResponse(config, initialCount, initialLastText, provider, timeoutMs = 150000, requestId = null) {
      const startTime = Date.now();
      let hasStreamingStarted = false;
      let lastContentLength = 0;
      let quietPeriods = 0;

      // Keepalive ping to background script every 2s to reset MV3 idle timers
      const keepAliveTimer = setInterval(() => {
        try {
          chrome.runtime.sendMessage({ action: "keepalive", request_id: requestId });
        } catch (_) {}
      }, 2000);

      return new Promise((resolve) => {
        let isDone = false;

        const finish = (result) => {
          if (isDone) return;
          isDone = true;
          clearInterval(keepAliveTimer);
          clearInterval(interval);
          resolve(result);
        };

        const interval = setInterval(async () => {
          const elapsed = Date.now() - startTime;
          if (elapsed > timeoutMs) {
            const currentResp = await getLatestResponse(config, provider, initialCount, initialLastText);
            finish(currentResp || "[aivco-kernel] Execution timed out waiting for AI response.");
            return;
          }

          // Check Rate Limit banner
          const rateLimitRegex = /(reached your .*limit|message limit|try again after|upgrade to claude pro|out of free messages|too many requests|usage limit)/i;
          const pageText = document.body ? (document.body.innerText || '') : '';
          if (rateLimitRegex.test(pageText)) {
            const banner = Array.from(document.querySelectorAll('div, p, span, h2, h3, [role="alert"]')).find(el => {
              const t = (el.innerText || '').trim();
              return t.length < 350 && rateLimitRegex.test(t);
            });
            if (banner) {
              finish({ isError: true, error: "RATE_LIMIT_REACHED" });
              return;
            }
          }

          const stopBtn = findElement(config.stopSelectors);
          const currentElements = getAssistantMessages(config);
          const currentCount = currentElements.length;

          // Inspect latest assistant message text
          const latestEl = currentElements.length > 0 ? currentElements[currentElements.length - 1] : null;
          const currentText = latestEl ? (latestEl.innerText || "").trim() : "";
          const isNewContent = (currentCount > initialCount) || (currentText.length > 20 && currentText !== initialLastText && !currentText.startsWith(initialLastText));

          // Check if generation is actively streaming
          if (stopBtn) {
            hasStreamingStarted = true;
            quietPeriods = 0;
            return;
          }

          // If streaming started and now stop button is gone:
          if (hasStreamingStarted && !stopBtn) {
            quietPeriods++;
            // 2 quiet checks (1 second) after stop button disappears
            if (quietPeriods >= 2) {
              setTimeout(async () => {
                const resp = await getLatestResponse(config, provider, initialCount, initialLastText);
                finish(resp);
              }, 300);
              return;
            }
          }

          // Quiet period detection for new content (covers cases where stop button wasn't detected)
          if (isNewContent && currentText.length > 20) {
            if (currentText.length === lastContentLength) {
              quietPeriods++;
              // 5 quiet checks (2.5 seconds) without length change indicates complete generation
              if (quietPeriods >= 5) {
                setTimeout(async () => {
                  const resp = await getLatestResponse(config, provider, initialCount, initialLastText);
                  finish(resp);
                }, 300);
                return;
              }
            } else {
              lastContentLength = currentText.length;
              quietPeriods = 0;
            }
          }
        }, 500);
      });
    }

    async function getLatestResponse(config, provider, initialCount = 0, initialLastText = "") {
      let mainText = "";

      // 1. Check primary response selectors (strictly ignore user bubbles)
      const assistantNodes = getAssistantMessages(config);
      if (assistantNodes.length > 0) {
        const lastEl = assistantNodes[assistantNodes.length - 1];
        const md = cleanAndFormatElement(lastEl);
        if (md && md.length > 10) {
          mainText = md;
        }
      }

      // 2. Check and append Artifact code if on Claude
      if (provider === "claude") {
        try {
          const artifactCode = await safeExtractClaudeArtifactCode();
          if (artifactCode && artifactCode.trim().length > 20) {
            const codeSnippet = artifactCode.replace(/^```\w*\n/, '').slice(0, 50).trim();
            if (!codeSnippet || !mainText.includes(codeSnippet)) {
              mainText = (mainText ? mainText + "\n\n" : "") + artifactCode;
            }
          }
        } catch (err) {
          console.warn("[AIVCO Bridge] Error extracting Claude artifact:", err);
        }
      }

      // If text is identical to initialLastText and count hasn't grown, do not replay stale response
      if (initialLastText && assistantNodes.length <= initialCount && mainText.trim() === initialLastText.trim()) {
        return "";
      }

      // Remove executive brief if echoed and unwanted header prefixes
      mainText = mainText.replace(/\[AIVCO EXECUTIVE BRIEF\][\s\S]*?\[\/AIVCO EXECUTIVE BRIEF\]/gi, '');
      mainText = mainText.replace(/^(?:Claude responded:?\s*)?/i, '');
      return mainText.trim();
    }

    // ── Message Listener & Asynchronous Push Return ──────────────────────────────

    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.action === "execute_prompt" || request.action === "dispatch") {
        // Immediately acknowledge receipt so the caller port doesn't dangle
        sendResponse({ received: true, request_id: request.request_id });

        // Run full async execution independently
        handleExecutePrompt(request);

        return false;
      }
    });

    async function handleExecutePrompt(request) {
      const requestId = request.request_id;
      const detected = detectProvider();
      const rawProv = (request.provider || detected || "").toLowerCase();
      const provider = (rawProv === "gemini_web" || rawProv === "gemini-web") ? "gemini" : (detected !== "unknown" ? detected : rawProv);
      const config = CONFIGS[provider] || CONFIGS[detected];

      if (!config) {
        const errPayload = {
          action: "dispatch_result",
          request_id: requestId,
          provider: provider,
          status: "error",
          error: `Unsupported provider or unrecognized page: ${window.location.hostname}`
        };
        chrome.runtime.sendMessage(errPayload);
        fetch("http://127.0.0.1:8000/api/bridge/result", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(errPayload)
        }).catch(() => {});
        return;
      }

      let promptText = request.prompt;

      // Force inline markdown code blocks for Claude to prevent artifact sandbox trap
      const codeKeywords = ["code", "html", "css", "javascript", "script", "website", "dashboard", "component", "app", "page", "react", "vue", "create", "build", "generate", "write a", "game"];
      const isCodePrompt = codeKeywords.some(kw => promptText.toLowerCase().includes(kw));
      if (provider === "claude" && isCodePrompt && !promptText.includes("IMPORTANT FORMATTING INSTRUCTION")) {
        promptText += "\n\n[IMPORTANT FORMATTING INSTRUCTION: Do NOT use the Artifacts tool, do NOT create a separate file/artifact card, and do NOT use the canvas. Output ALL code directly inline inside this chat message using standard markdown fenced code blocks like ```html ... ```.]";
      }

      try {
        const inputEl = await waitForEditor(config.inputSelectors, 12000);
        if (!inputEl) {
          throw new Error(`Could not find prompt input field on ${provider} after waiting 12s. Please verify you are logged into your account.`);
        }

        const initialElements = getAssistantMessages(config);
        const initialCount = request.baseline ? request.baseline.baselineMsgCount : initialElements.length;
        const initialLastText = request.baseline ? request.baseline.baselineLastMsgText : (initialElements.length > 0 ? (initialElements[initialElements.length - 1].innerText || "").trim() : "");

        if (provider === "claude") {
          await submitClaudePrompt(inputEl, promptText);
        } else {
          await submitGenericPrompt(inputEl, promptText, config);
        }

        const resp = await waitForResponse(config, initialCount, initialLastText, provider, 150000, requestId);

        let donePayload;
        if (resp && resp.isError) {
          donePayload = {
            action: "dispatch_result",
            request_id: requestId,
            provider: provider,
            status: "error",
            markdown: "",
            error: resp.error || "GENERATION_ERROR"
          };
        } else {
          donePayload = {
            action: "dispatch_result",
            request_id: requestId,
            provider: provider,
            status: "done",
            markdown: typeof resp === "string" ? resp : (resp ? resp.markdown || "" : "")
          };
        }

        // 1. Push to background.js
        chrome.runtime.sendMessage(donePayload);

        // 2. Direct HTTP POST callback to server
        fetch("http://127.0.0.1:8000/api/bridge/result", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(donePayload)
        }).catch(() => {});
      } catch (err) {
        console.error("[AIVCO Bridge] Error executing prompt:", err);
        let fallbackText = "";
        try {
          fallbackText = await getLatestResponse(config, provider, initialCount, initialLastText);
        } catch (_) {}

        const errPayload = {
          action: "dispatch_result",
          request_id: requestId,
          provider: provider,
          status: fallbackText ? "done" : "error",
          markdown: fallbackText,
          error: fallbackText ? null : (err.message || String(err))
        };

        chrome.runtime.sendMessage(errPayload);
        fetch("http://127.0.0.1:8000/api/bridge/result", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(errPayload)
        }).catch(() => {});
      }
    }

  })();
}
