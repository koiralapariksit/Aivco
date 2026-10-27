
    const { useState, useEffect, useRef, useMemo, useCallback } = React;
    const API_BASE = window.location.origin;

    const THEMES = ['argon', 'ultron', 'amber'];
    const THEME_NAMES = { argon: 'ARGON', ultron: 'ULTRON', amber: 'AMBER' };
    const THEME_COLORS = { argon: '#56E2F5', ultron: '#FF4455', amber: '#F59E0B' };

    // ── Universal Reliable Clipboard Function ───────────────────────────────
    function copyToClipboard(text) {
      if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(text).catch(() => {
          fallbackCopyText(text);
        });
      } else {
        fallbackCopyText(text);
        return Promise.resolve();
      }
    }

    function fallbackCopyText(text) {
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.position = "fixed";
      textArea.style.left = "-999999px";
      textArea.style.top = "-999999px";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      try {
        document.execCommand('copy');
      } catch (err) {
        console.error('Fallback copy failed', err);
      }
      document.body.removeChild(textArea);
    }

    // ── Utility: Tokenize markdown into text and code blocks ────────────────
    function parseMessageTokens(content) {
      if (!content) return [];
      const codeBlockRegex = /```([a-zA-Z0-9_\-+]*)\n([\s\S]*?)```/g;
      const parts = [];
      let lastIndex = 0;
      let match;

      while ((match = codeBlockRegex.exec(content)) !== null) {
        if (match.index > lastIndex) {
          parts.push({
            type: 'text',
            content: content.slice(lastIndex, match.index)
          });
        }
        parts.push({
          type: 'code',
          lang: match[1] || 'text',
          content: match[2]
        });
        lastIndex = match.index + match[0].length;
      }

      if (lastIndex < content.length) {
        parts.push({
          type: 'text',
          content: content.slice(lastIndex)
        });
      }

      return parts;
    }

    // ── CodeBlock Component with Syntax Highlighting & Action Bar ────────────
    function CodeBlock({ language, code, onOpenArtifact }) {
      const [copied, setCopied] = useState(false);
      const [wrapped, setWrapped] = useState(false);
      const codeRef = useRef(null);

      const lang = (language || 'code').toLowerCase();
      const lineCount = code ? code.trim().split('\n').length : 0;
      const isWebArtifact = ['html', 'xml', 'svg', 'javascript', 'js', 'css'].includes(lang) ||
                            code.includes('<!DOCTYPE') || code.includes('<html') || code.includes('<svg') || lineCount >= 10;

      useEffect(() => {
        if (codeRef.current && window.hljs) {
          try {
            window.hljs.highlightElement(codeRef.current);
          } catch (e) {
            // fallback
          }
        }
      }, [code, language]);

      const handleCopy = () => {
        copyToClipboard(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      };

      const handleDownload = () => {
        const extMap = {
          html: 'html', python: 'py', py: 'py', javascript: 'js', js: 'js',
          typescript: 'ts', ts: 'ts', css: 'css', json: 'json', markdown: 'md',
          md: 'md', bash: 'sh', sh: 'sh', sql: 'sql', rust: 'rs'
        };
        const ext = extMap[lang] || 'txt';
        const blob = new Blob([code], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `aivco_artifact_${Date.now()}.${ext}`;
        a.click();
        URL.revokeObjectURL(url);
      };

      return (
        <div className="my-2.5 rounded border border-border bg-[#070B12] overflow-hidden text-xs font-mono shadow-xl chat-selectable">
          {/* Top Bar */}
          <div className="flex items-center justify-between px-3 py-1.5 bg-[#0E1724] border-b border-border text-[11px] select-none">
            <div className="flex items-center gap-2">
              <span className="font-bold text-argon uppercase tracking-wider">{lang}</span>
              <span className="text-slate-600">•</span>
              <span className="text-slate-400">{lineCount} lines</span>
            </div>
            <div className="flex items-center gap-1.5">
              {isWebArtifact && (
                <button
                  type="button"
                  onClick={() => onOpenArtifact && onOpenArtifact({ lang, code, title: `${lang.toUpperCase()} Artifact` })}
                  className="px-2 py-0.5 rounded bg-argon/15 border border-argon/50 text-argon hover:bg-argon/30 transition-colors flex items-center gap-1 font-bold shadow-[0_0_8px_rgba(86,226,245,0.25)]"
                  title="Open Live HTML Preview Drawer"
                >
                  <span>⚡ LIVE PREVIEW ⤢</span>
                </button>
              )}
              <button
                type="button"
                onClick={() => setWrapped(!wrapped)}
                className={`px-1.5 py-0.5 rounded border transition-colors ${wrapped ? 'border-argon text-argon bg-argon/10' : 'border-border text-slate-400 hover:text-slate-200'}`}
                title="Toggle Wrap"
              >
                {wrapped ? 'WRAP' : 'NO-WRAP'}
              </button>
              <button
                type="button"
                onClick={handleDownload}
                className="px-1.5 py-0.5 rounded border border-border text-slate-400 hover:text-slate-200 hover:border-slate-500 transition-colors"
                title="Download Snippet"
              >
                ⬇
              </button>
              <button
                type="button"
                onClick={handleCopy}
                className={`px-2 py-0.5 rounded border transition-colors font-bold ${copied ? 'border-emerald-500 text-emerald-400 bg-emerald-500/10' : 'border-border text-slate-300 hover:border-argon'}`}
              >
                {copied ? '✓ COPIED' : '📋 COPY'}
              </button>
            </div>
          </div>
          {/* Code Viewer */}
          <pre className={`p-3 overflow-x-auto text-[11.5px] leading-relaxed m-0 chat-selectable ${wrapped ? 'whitespace-pre-wrap' : 'whitespace-pre'}`}>
            <code ref={codeRef} className={`language-${lang} font-mono chat-selectable`}>
              {code}
            </code>
          </pre>
        </div>
      );
    }

    // ── Markdown Message Renderer ────────────────────────────────────────────
    function FormattedMessage({ text, onOpenArtifact }) {
      const tokens = useMemo(() => parseMessageTokens(text), [text]);

      return (
        <div className="w-full chat-selectable">
          {tokens.map((token, idx) => {
            if (token.type === 'code') {
              return (
                <CodeBlock 
                  key={idx} 
                  language={token.lang} 
                  code={token.content} 
                  onOpenArtifact={onOpenArtifact} 
                />
              );
            }
            const html = window.marked ? window.marked.parse(token.content) : token.content;
            return (
              <div 
                key={idx} 
                className="markdown-body chat-selectable" 
                dangerouslySetInnerHTML={{ __html: html }} 
              />
            );
          })}
        </div>
      );
    }

    // ── 3D HTML5 Canvas Memory Constellation Sphere ─────────────────────────
    function ConstellationSphere({ themeColor = '#56E2F5' }) {
      const canvasRef = useRef(null);
      const isDraggingRef = useRef(false);
      const lastMouseRef = useRef({ x: 0, y: 0 });
      const rotRef = useRef({ x: 0.35, y: 0.5 });

      useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let animId;

        const numPoints = 36;
        const points = [];
        const phi = Math.PI * (3 - Math.sqrt(5));

        for (let i = 0; i < numPoints; i++) {
          const y = 1 - (i / (numPoints - 1)) * 2;
          const radius = Math.sqrt(1 - y * y);
          const theta = phi * i;
          const x = Math.cos(theta) * radius;
          const z = Math.sin(theta) * radius;
          points.push({ x, y, z });
        }

        const render = () => {
          if (!isDraggingRef.current) {
            rotRef.current.y += 0.006;
            rotRef.current.x += 0.002;
          }

          ctx.clearRect(0, 0, canvas.width, canvas.height);
          const cx = canvas.width / 2;
          const cy = canvas.height / 2;
          const sphereRadius = 46;

          const cosX = Math.cos(rotRef.current.x);
          const sinX = Math.sin(rotRef.current.x);
          const cosY = Math.cos(rotRef.current.y);
          const sinY = Math.sin(rotRef.current.y);

          const rotated = points.map(p => {
            let x1 = p.x * cosY - p.z * sinY;
            let z1 = p.z * cosY + p.x * sinY;
            let y1 = p.y * cosX - z1 * sinX;
            let z2 = z1 * cosX + p.y * sinX;

            const fov = 160;
            const scale = fov / (fov + z2 * sphereRadius);
            const px = cx + x1 * sphereRadius * scale;
            const py = cy + y1 * sphereRadius * scale;

            return { x: px, y: py, z: z2, scale };
          });

          for (let i = 0; i < rotated.length; i++) {
            for (let j = i + 1; j < rotated.length; j++) {
              const dx = rotated[i].x - rotated[j].x;
              const dy = rotated[i].y - rotated[j].y;
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist < 32) {
                const alpha = Math.max(0.04, (1 - dist / 32) * 0.35);
                ctx.beginPath();
                ctx.moveTo(rotated[i].x, rotated[i].y);
                ctx.lineTo(rotated[j].x, rotated[j].y);
                ctx.strokeStyle = themeColor;
                ctx.globalAlpha = alpha;
                ctx.lineWidth = 0.75;
                ctx.stroke();
              }
            }
          }

          for (let i = 0; i < rotated.length; i++) {
            const pt = rotated[i];
            const alpha = Math.max(0.2, (pt.z + 1) / 2);
            const r = Math.max(1.2, 2.2 * pt.scale);

            ctx.beginPath();
            ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
            ctx.fillStyle = themeColor;
            ctx.globalAlpha = alpha;
            ctx.shadowColor = themeColor;
            ctx.shadowBlur = 5;
            ctx.fill();
            ctx.shadowBlur = 0;
          }
          ctx.globalAlpha = 1.0;

          animId = requestAnimationFrame(render);
        };

        render();
        return () => cancelAnimationFrame(animId);
      }, [themeColor]);

      const handleMouseDown = (e) => {
        isDraggingRef.current = true;
        lastMouseRef.current = { x: e.clientX, y: e.clientY };
      };

      const handleMouseMove = (e) => {
        if (!isDraggingRef.current) return;
        const dx = e.clientX - lastMouseRef.current.x;
        const dy = e.clientY - lastMouseRef.current.y;
        rotRef.current.y += dx * 0.01;
        rotRef.current.x -= dy * 0.01;
        lastMouseRef.current = { x: e.clientX, y: e.clientY };
      };

      const handleMouseUp = () => {
        isDraggingRef.current = false;
      };

      return (
        <div 
          className="relative flex flex-col items-center justify-center p-2 rounded bg-card/60 border border-border overflow-hidden cursor-grab active:cursor-grabbing select-none"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <div className="w-full flex items-center justify-between text-[10px] font-mono text-slate-400 mb-0.5">
            <span>3D MEMORY CONSTELLATION</span>
            <span className="text-argon animate-pulse">V2 GRAPH</span>
          </div>
          <canvas ref={canvasRef} width="220" height="120" className="w-full h-[120px]" />
          <div className="text-[9px] font-mono text-slate-500 mt-0.5">Drag to rotate • 36 Nodes Synced</div>
        </div>
      );
    }

    // ── Main Dashboard Application ──────────────────────────────────────────
    function App() {
      // Theme & Telemetry
      const [theme, setTheme] = useState('argon');
      const [health, setHealth] = useState({ kernel: true, tier1_local: false, tier2_groq: true, tier3_gemini: true, brain4_web: true, active_override: null });
      const [state, setState] = useState({
        active_project: 'aivco',
        projects: { aivco: [['Architecture', 100], ['Kernel Resolver', 95], ['Vector FAISS', 25], ['Claude Chat Engine', 100]] },
        layers: { Semantic: [], Episodic: [], Procedural: [], Creative: [] },
        tasks: [],
        dataset_stats: { total_entries: 0, sources: {} },
        core_profile: {}
      });

      // View & Modals
      const [activeView, setActiveView] = useState('Chat');
      const [showNewEngramModal, setShowNewEngramModal] = useState(false);
      const [showSearchModal, setShowSearchModal] = useState(false);
      const [searchQuery, setSearchQuery] = useState('');
      const [showHudRing, setShowHudRing] = useState(true);

      // HUD Toast Notifications State
      const [toasts, setToasts] = useState([]);
      const [isLaunchingBrowser, setIsLaunchingBrowser] = useState(false);

      // Live Browser Bridge (CDP) State
      const [cdpStatus, setCdpStatus] = useState({ running: false, port: 9222 });

      const checkCdpStatus = useCallback(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/browser/cdp_status`);
          if (res.ok) {
            const data = await res.json();
            setCdpStatus(data);
          }
        } catch (_) {}
      }, []);

      useEffect(() => {
        checkCdpStatus();
        const iv = setInterval(checkCdpStatus, 5000);
        return () => clearInterval(iv);
      }, [checkCdpStatus]);

      const handleLaunchCdp = async () => {
        try {
          showToast("Launching Persistent Agent Browser (Port 9222) with Claude, ChatGPT, Gemini & Grok...", "info", 4000);
          const res = await fetch(`${API_BASE}/api/browser/launch_cdp`, { method: "POST" });
          const data = await res.json();
          showToast(data.message || "Agent Browser active on port 9222. Log into your accounts directly in the browser.", "success", 7000);
          checkCdpStatus();
          fetchVaultStatus();
        } catch (err) {
          showToast(`Browser Bridge launch failed: ${err.message}`, "error", 5000);
        }
      };

      // AI Credentials Vault Modal State
      const [showVaultModal, setShowVaultModal] = useState(false);
      const [vaultData, setVaultData] = useState(null);
      const [vaultLoading, setVaultLoading] = useState(false);
      const [launchingSlot, setLaunchingSlot] = useState(null);

      const showToast = useCallback((message, type = 'info', duration = 4000) => {
        const id = Date.now() + Math.random();
        setToasts(prev => [...prev, { id, message, type }]);
        setTimeout(() => {
          setToasts(prev => prev.filter(t => t.id !== id));
        }, duration);
      }, []);

      useEffect(() => {
        window.showToast = showToast;
      }, [showToast]);

      const fetchVaultStatus = async () => {
        try {
          setVaultLoading(true);
          const res = await fetch(`${API_BASE}/api/vault/status`);
          if (res.ok) {
            const data = await res.json();
            setVaultData(data);
          }
        } catch (e) {
          console.error("Vault fetch error:", e);
        } finally {
          setVaultLoading(false);
        }
      };

      const handleVaultLogin = async (provider, slot = null) => {
        const slotKey = slot ? `${provider}-${slot}` : provider;
        try {
          setLaunchingSlot(slotKey);
          showToast(`Launching ${provider.toUpperCase()}${slot ? ` Slot #${slot}` : ''} login window...`, 'info', 3000);
          const res = await fetch(`${API_BASE}/api/vault/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, slot })
          });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          showToast('Browser window opened. Log in to your account, then CLOSE that browser window when finished so AIVCO can connect.', 'success', 8000);
          setTimeout(fetchVaultStatus, 4000);
          setTimeout(fetchVaultStatus, 12000);
        } catch (err) {
          showToast(`Failed to open login: ${err.message}`, 'error', 5000);
        } finally {
          setLaunchingSlot(null);
        }
      };

      const handleSetActiveSlot = async (slotNum) => {
        try {
          const res = await fetch(`${API_BASE}/api/vault/set_active_slot`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ slot: slotNum })
          });
          if (res.ok) {
            showToast(`Active Claude slot switched to Slot #${slotNum}`, 'success', 3000);
            fetchVaultStatus();
          }
        } catch (err) {
          showToast(`Error switching slot: ${err.message}`, 'error', 4000);
        }
      };

      // 4-Brain Target Selector State
      // Options: 'auto', 'kernel', 'groq', 'gemini', 'web_search', 'claude_web', 'chatgpt_web', 'grok_web'
      const [selectedBrain, setSelectedBrain] = useState('auto');

      const [uptime, setUptime] = useState(0);
      const [hubStatus, setHubStatus] = useState('idle');
      const [lastReply, setLastReply] = useState({ provider: 'system', latency: 4, src: 'Kernel' });

      // Chat & Artifact State
      const [messages, setMessages] = useState([
        {
          id: 1,
          role: 'assistant',
          text: '**AIVCO // ARGON EDITION online.** 4-Brain Cognitive Architecture initialized (Stage 0/1 Kernel, Tier 1 Local, Tier 2 Groq, Tier 3 Gemini, Brain 4 Web/Browser). Ready for tactical instructions, Sir.',
          provider: 'aivco-kernel',
          tier: 0,
          src: 'Kernel',
          lat: 3
        }
      ]);
      const [inputVal, setInputVal] = useState('');
      const [attachments, setAttachments] = useState([]);
      const [isSending, setIsSending] = useState(false);
      const [copiedId, setCopiedId] = useState(null);
      const [approvedPairs, setApprovedPairs] = useState({});

      // Claude-style Artifact Drawer
      const [currentArtifact, setCurrentArtifact] = useState(null);
      const [artifactTab, setArtifactTab] = useState('preview'); // 'preview' | 'code'
      const [isArtifactFullscreen, setIsArtifactFullscreen] = useState(false);
      const [iframeKey, setIframeKey] = useState(0);

      // Refs
      const chatEndRef = useRef(null);
      const textareaRef = useRef(null);
      const fileInputRef = useRef(null);

      // Quick forms
      const [newEngram, setNewEngram] = useState({ layer: 'Semantic', title: '', body: '' });
      const [newTaskText, setNewTaskText] = useState('');

      // ── Effects ────────────────────────────────────────────────────────────

      useEffect(() => {
        document.body.setAttribute('data-theme', theme);
      }, [theme]);

      useEffect(() => {
        const timer = setInterval(() => setUptime(u => u + 1), 1000);
        return () => clearInterval(timer);
      }, []);

      const fetchHealth = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/health`);
          if (res.ok) {
            const data = await res.json();
            setHealth({
              kernel: data.kernel !== undefined ? data.kernel : true,
              tier1_local: data.tier1_local !== undefined ? data.tier1_local : (data.tiers?.local_llama?.ready || false),
              tier2_groq: data.tier2_groq !== undefined ? data.tier2_groq : (data.tiers?.groq?.ready || false),
              tier3_gemini: data.tier3_gemini !== undefined ? data.tier3_gemini : (data.tiers?.gemini?.ready || false),
              brain4_web: data.brain4_web !== undefined ? data.brain4_web : true,
              active_override: data.active_override || null
            });
          }
        } catch (e) {
          console.error("Health check error:", e);
        }
      };

      const fetchState = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/state`);
          if (res.ok) {
            const data = await res.json();
            setState(data);
          }
        } catch (e) {
          console.error("State fetch error:", e);
        }
      };

      useEffect(() => {
        fetchHealth();
        fetchState();
        const poll = setInterval(() => {
          fetchHealth();
          fetchState();
        }, 5000);
        return () => clearInterval(poll);
      }, []);

      useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, [messages]);

      // Keyboard shortcut (Ctrl+K)
      useEffect(() => {
        const handleKeyDown = (e) => {
          if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            setShowSearchModal(s => !s);
          }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
      }, []);

      // ── Actions ────────────────────────────────────────────────────────────

      const cycleTheme = () => {
        const idx = THEMES.indexOf(theme);
        setTheme(THEMES[(idx + 1) % THEMES.length]);
      };

      const handleCopyMessage = (msgId, text) => {
        copyToClipboard(text);
        setCopiedId(msgId);
        setTimeout(() => setCopiedId(null), 2000);
      };

      const handleSendChat = async (overridePrompt, explicitTarget) => {
        const prompt = (overridePrompt !== undefined ? overridePrompt : inputVal).trim();
        if ((!prompt && attachments.length === 0) || isSending) return;

        setInputVal('');
        const userMsg = { 
          id: Date.now(), 
          role: 'user', 
          text: prompt || (attachments.length > 0 ? `Attached ${attachments.length} files.` : ''),
          attachments: [...attachments]
        };
        setMessages(m => [...m, userMsg]);
        setIsSending(true);
        setHubStatus('processing');

        if (textareaRef.current) {
          textareaRef.current.style.height = 'auto';
        }

        const modelTarget = explicitTarget || selectedBrain || 'auto';

        try {
          const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              prompt: prompt || 'Analyze attached files',
              target_model: modelTarget,
              auto_route: modelTarget === 'auto',
              attachments: attachments.map(a => ({ name: a.name, content: a.content }))
            })
          });

          setAttachments([]);

          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();

          const botMsg = {
            id: Date.now() + 1,
            role: 'assistant',
            text: data.response || '[No response payload]',
            provider: data.provider || 'unknown',
            tier: data.tier !== undefined ? data.tier : 0,
            src: data.src || 'Kernel',
            lat: data.lat || 10,
            web_searched: !!data.web_searched
          };

          setMessages(m => [...m, botMsg]);
          setLastReply({ provider: data.provider, latency: data.lat, src: data.src });
          fetchState();

          // Auto-open artifact if response contains HTML document or full code
          if (data.response && (data.response.includes('<!DOCTYPE html') || data.response.includes('<html'))) {
            const htmlMatch = data.response.match(/```(?:html)?\n([\s\S]*?)```/i);
            if (htmlMatch && htmlMatch[1]) {
              setCurrentArtifact({
                title: 'Live Web Artifact',
                lang: 'html',
                code: htmlMatch[1]
              });
              setArtifactTab('preview');
            }
          }

        } catch (err) {
          setMessages(m => [...m, {
            id: Date.now() + 1,
            role: 'assistant',
            text: `**[Communication Error]** ${err.message}`,
            provider: 'error',
            tier: 0,
            src: 'System',
            lat: 0
          }]);
        } finally {
          setIsSending(false);
          setHubStatus('idle');
        }
      };

      // Auto-Paste Smart Detection (>500 chars or >15 lines)
      const handlePaste = (e) => {
        const pasted = e.clipboardData.getData('text');
        if (pasted && (pasted.length > 500 || pasted.split('\n').length > 15)) {
          e.preventDefault();
          const count = attachments.length + 1;
          setAttachments(prev => [
            ...prev,
            {
              id: Date.now(),
              name: `pasted_snippet_${count}.txt`,
              content: pasted,
              size: Math.round(pasted.length / 1024 * 10) / 10,
              lines: pasted.split('\n').length
            }
          ]);
        }
      };

      // Native File Upload
      const handleFileUpload = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
          const content = evt.target.result;
          setAttachments(prev => [
            ...prev,
            {
              id: Date.now(),
              name: file.name,
              content: content,
              size: Math.round(content.length / 1024 * 10) / 10,
              lines: content.split('\n').length
            }
          ]);
        };
        reader.readAsText(file);
        e.target.value = '';
      };

      // Approve Golden Pair for Fine-Tuning
      const handleApprovePair = async (botMsgId, userPrompt, responseText) => {
        try {
          const res = await fetch(`${API_BASE}/api/dataset/approve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              prompt: userPrompt,
              response: responseText,
              source: 'golden_user_approved',
              task_type: 'chat'
            })
          });
          if (res.ok) {
            const data = await res.json();
            setApprovedPairs(p => ({ ...p, [botMsgId]: true }));
            setState(s => ({
              ...s,
              dataset_stats: { ...s.dataset_stats, total_entries: data.total }
            }));
          }
        } catch (e) {
          console.error("Approve pair error:", e);
        }
      };

      const handleToggleTask = async (task) => {
        try {
          const res = await fetch(`${API_BASE}/api/tasks/toggle`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: task.text, project: task.project })
          });
          if (res.ok) fetchState();
        } catch (e) {
          console.error('Toggle task error:', e);
        }
      };

      const handleAddTask = async (e) => {
        e?.preventDefault();
        if (!newTaskText.trim()) return;
        try {
          const res = await fetch(`${API_BASE}/api/tasks/add`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: newTaskText.trim(), project: state.active_project })
          });
          if (res.ok) {
            setNewTaskText('');
            fetchState();
          }
        } catch (e) {
          console.error('Add task error:', e);
        }
      };

      const handleAddEngram = async (e) => {
        e.preventDefault();
        if (!newEngram.title.trim() && !newEngram.body.trim()) return;
        try {
          const res = await fetch(`${API_BASE}/api/memory/add`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newEngram)
          });
          if (res.ok) {
            setNewEngram({ layer: 'Semantic', title: '', body: '' });
            setShowNewEngramModal(false);
            fetchState();
          }
        } catch (e) {
          console.error('Add engram error:', e);
        }
      };

      const handleCreateProject = async () => {
        const name = prompt('Enter new project name:');
        if (!name || !name.trim()) return;
        try {
          const res = await fetch(`${API_BASE}/api/projects/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name.trim() })
          });
          if (res.ok) fetchState();
        } catch (e) {
          console.error('Create project error:', e);
        }
      };

      const formatUptime = (sec) => {
        const h = Math.floor(sec / 3600).toString().padStart(2, '0');
        const m = Math.floor((sec % 3600) / 60).toString().padStart(2, '0');
        const s = (sec % 60).toString().padStart(2, '0');
        return `${h}:${m}:${s}`;
      };

      // ── Render ─────────────────────────────────────────────────────────────

      return (
        <div className="flex flex-col h-screen overflow-hidden bg-obsidian text-slate-200">
          {/* HUD Toast Notifications Container */}
          <div className="fixed top-14 right-4 z-[9999] flex flex-col gap-2.5 pointer-events-none max-w-sm w-full">
            {toasts.map(t => (
              <div
                key={t.id}
                className={`pointer-events-auto flex items-start gap-2.5 px-4 py-3 rounded-lg border backdrop-blur-md shadow-2xl transition-all text-xs font-mono animate-slide-in ${
                  t.type === 'success'
                    ? 'bg-[#0A101A]/95 border-[#00ff88] text-[#00ff88] shadow-[0_0_15px_rgba(0,255,136,0.3)]'
                    : t.type === 'error'
                    ? 'bg-[#0A101A]/95 border-rose-500 text-rose-300 shadow-[0_0_15px_rgba(244,63,94,0.3)]'
                    : 'bg-[#0A101A]/95 border-[#00f0ff] text-[#00f0ff] shadow-[0_0_15px_rgba(0,240,255,0.3)]'
                }`}
              >
                <span className="text-sm mt-0.5">
                  {t.type === 'success' ? '✓' : t.type === 'error' ? '⚠️' : '⚡'}
                </span>
                <div className="flex-1 leading-relaxed text-slate-100 font-sans">
                  {t.message}
                </div>
                <button
                  type="button"
                  onClick={() => setToasts(prev => prev.filter(x => x.id !== t.id))}
                  className="text-slate-400 hover:text-slate-100 text-xs font-bold ml-1.5"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>

          
          {/* Top Bar: Live Status Badges for all 4 Brains */}
          <header className="h-12 border-b border-border bg-surface/90 px-4 flex items-center justify-between z-20 text-xs font-mono select-none">
            {/* Left: Branding & Core Mode */}
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 rounded border border-argon flex items-center justify-center font-bold text-argon shadow-[0_0_10px_rgba(86,226,245,0.3)]">
                A
              </div>
              <span className="font-bold tracking-widest text-slate-100">AIVCO // ARGON</span>
              <span className="text-slate-600">|</span>
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-argon/10 border border-argon/30 text-argon">
                <span className="w-1.5 h-1.5 rounded-full bg-argon animate-pulse"></span>
                <span>4-BRAIN ARCHITECTURE</span>
              </div>
            </div>

            {/* Center: Live 4-Brain Readiness Badges */}
            <div className="flex items-center gap-3 hidden md:flex text-[11px]">
              <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-900 border border-border">
                <span className="w-2 h-2 rounded-full bg-argon"></span>
                <span className="text-slate-300">Brain 1: Kernel</span>
              </div>
              <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-900 border border-border/50 opacity-40" title="Local Brain 2 disabled to conserve system RAM">
                <span className="w-2 h-2 rounded-full bg-slate-600"></span>
                <span className="text-slate-400 font-mono">T1: Disabled (RAM Safe)</span>
              </div>
              <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-900 border border-border">
                <span className={`w-2 h-2 rounded-full ${health.tier2_groq ? 'bg-amber-400' : 'bg-rose-500'}`}></span>
                <span className="text-slate-300">T2: Groq 70B</span>
              </div>
              <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-900 border border-border">
                <span className={`w-2 h-2 rounded-full ${health.tier3_gemini ? 'bg-blue-400' : 'bg-rose-500'}`}></span>
                <span className="text-slate-300">T3: Gemini Pro</span>
              </div>
              <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-900 border border-border">
                <span className="w-2 h-2 rounded-full bg-purple-400"></span>
                <span className="text-slate-300">Brain 4: Web/Browser</span>
              </div>
            </div>

            {/* Right: Controls & Themes */}
            <div className="flex items-center gap-2.5">
              <button 
                id="connectWebAccountsBtn"
                type="button"
                onClick={() => {
                  setShowVaultModal(true);
                  checkCdpStatus();
                  fetchVaultStatus();
                }}
                className="px-3 py-1 rounded border border-argon/70 text-argon bg-argon/10 hover:bg-argon/25 active:scale-95 cursor-pointer transition-all flex items-center gap-1.5 font-bold shadow-[0_0_12px_rgba(86,226,245,0.25)] select-none"
                style={{ cursor: 'pointer', pointerEvents: 'auto', zIndex: 1000 }}
                title="AIVCO Live Browser Bridge (Persistent Port 9222 CDP)"
              >
                <span>🌐</span>
                <span>Live Browser Bridge</span>
                <span className={`text-[10px] px-1.5 py-0.2 rounded font-bold border ${cdpStatus?.running ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40' : 'bg-amber-500/20 text-amber-300 border-amber-500/40'}`}>
                  {cdpStatus?.running ? '🟢 LIVE CDP' : '⚪ PORT 9222'}
                </span>
              </button>
              <button 
                type="button"
                onClick={() => setShowHudRing(!showHudRing)}
                className={`px-2 py-0.5 rounded border transition-colors ${showHudRing ? 'border-argon text-argon bg-argon/10' : 'border-slate-700 text-slate-400'}`}
                title="Toggle Vector Radar HUD"
              >
                {showHudRing ? 'RADAR: ON' : 'RADAR: OFF'}
              </button>

              <button 
                type="button"
                onClick={cycleTheme}
                className="px-2 py-0.5 rounded border border-border hover:border-argon transition-colors flex items-center gap-1 text-slate-300"
              >
                <span>CORE:</span>
                <span className="text-argon font-bold">{THEME_NAMES[theme]}</span>
              </button>

              <button 
                type="button"
                onClick={() => setShowSearchModal(true)}
                className="px-2 py-0.5 rounded border border-border bg-slate-900/50 hover:border-slate-500 text-slate-400 flex items-center gap-1"
                title="Global Search"
              >
                <span>CTRL+K</span>
              </button>
            </div>
          </header>

          {/* Main 3-Column Tactical Grid */}
          <div className="flex-1 grid grid-cols-12 gap-2 p-2 overflow-hidden relative">
            
            {/* ── LEFT SIDEBAR (Col 1-3): Clock Dial, Projects, Milestones, Memory ── */}
            <aside className="col-span-3 flex flex-col gap-2 overflow-y-auto pr-1">
              
              {/* Chrono Dial & Uptime */}
              <div className="glass-bracket p-3 rounded">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono tracking-wider text-slate-400 uppercase">SYS TELEMETRY // UPTIME</span>
                  <span className="text-xs font-mono text-argon">{formatUptime(uptime)}</span>
                </div>
                <div className="flex items-center gap-3">
                  <svg className="w-10 h-10 -rotate-90" viewBox="0 0 36 36">
                    <circle cx="18" cy="18" r="15" fill="none" stroke="#1A293E" strokeWidth="2.5" />
                    <circle 
                      cx="18" cy="18" r="15" fill="none" stroke="var(--primary)" strokeWidth="2.5"
                      strokeDasharray="100" strokeDashoffset={100 - ((uptime % 60) / 60) * 100}
                      strokeLinecap="round" className="transition-all duration-300"
                    />
                  </svg>
                  <div>
                    <div className="text-sm font-bold tracking-wide">COGNITIVE ENGINE 2.0</div>
                    <div className="text-[11px] font-mono text-emerald-400 flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
                      <span>ALL LAYERS SYNCED</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Active Project & 4 Circular Milestone SVG Gauges */}
              <div className="glass-bracket p-3 rounded flex flex-col gap-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-400 uppercase">PROJECT // MILESTONES</span>
                  <button 
                    type="button"
                    onClick={handleCreateProject}
                    className="text-argon hover:text-white px-1.5 py-0.5 rounded border border-argon/30 hover:border-argon"
                    title="Create Project"
                  >
                    + NEW
                  </button>
                </div>
                
                <select 
                  value={state.active_project} 
                  onChange={(e) => {
                    handleSendChat(`project use ${e.target.value}`);
                  }}
                  className="w-full bg-slate-900 border border-border rounded px-2 py-1 text-xs font-mono text-slate-200 outline-none focus:border-argon"
                >
                  {Object.keys(state.projects || {}).map(pid => (
                    <option key={pid} value={pid}>{pid.toUpperCase()} {pid === state.active_project ? '★' : ''}</option>
                  ))}
                </select>

                {/* 4 Circular SVG Gauges */}
                <div className="grid grid-cols-2 gap-2 mt-1">
                  {(state.projects[state.active_project] || []).slice(0, 4).map(([name, pct], i) => (
                    <div key={i} className="flex flex-col items-center p-2 rounded bg-slate-950/50 border border-border/50 text-center">
                      <div className="relative w-12 h-12 flex items-center justify-center mb-1">
                        <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                          <circle cx="18" cy="18" r="14" fill="none" stroke="#1A293E" strokeWidth="2.5" />
                          <circle 
                            cx="18" cy="18" r="14" fill="none" stroke="var(--primary)" strokeWidth="2.5"
                            strokeDasharray="88" strokeDashoffset={88 - (pct / 100) * 88}
                            strokeLinecap="round" className="transition-all duration-500"
                          />
                        </svg>
                        <span className="absolute text-[10px] font-mono font-bold text-slate-100">{pct}%</span>
                      </div>
                      <span className="text-[10px] font-mono text-slate-300 truncate w-full" title={name}>{name}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Expandable Memory Layers (Semantic, Episodic, Procedural, Creative) */}
              <div className="glass-bracket p-3 rounded flex-1 flex flex-col gap-2 min-h-[220px]">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-400 uppercase">MEMORY ENGRAMS</span>
                  <button 
                    type="button"
                    onClick={() => setShowNewEngramModal(true)}
                    className="text-[10px] text-argon hover:text-white px-1.5 py-0.5 rounded border border-argon/40"
                  >
                    + ENGRAM
                  </button>
                </div>

                <div className="flex-1 space-y-2 overflow-y-auto pr-1">
                  {['Semantic', 'Procedural', 'Episodic', 'Creative'].map((layerKey) => {
                    const items = state.layers?.[layerKey] || [];
                    return (
                      <div key={layerKey} className="border border-border/60 rounded bg-slate-950/40 p-2">
                        <div className="flex items-center justify-between text-xs font-mono text-slate-300 mb-1">
                          <span className="font-bold flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-sm bg-argon"></span>
                            {layerKey}
                          </span>
                          <span className="text-[10px] text-slate-500">[{items.length}]</span>
                        </div>
                        <div className="space-y-1 text-[11px] text-slate-400">
                          {items.slice(0, 2).map((item, idx) => (
                            <div key={idx} className="truncate hover:text-slate-200 cursor-pointer" title={item.body}>
                              • {item.title || item.body}
                            </div>
                          ))}
                          {items.length === 0 && (
                            <div className="text-[10px] text-slate-600 italic">No records in layer</div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </aside>

            {/* ── CENTER COMMAND DECK (Col 4-9): Hub + Chat Console & Live Artifacts ── */}
            <main className="col-span-6 flex flex-col gap-2 overflow-hidden">
              
              {/* Central Vector Hub (HUD Radar) */}
              {showHudRing && (
                <div className="glass-bracket p-3 rounded flex items-center justify-between h-28 relative overflow-hidden bg-gradient-to-r from-card to-surface transition-all">
                  
                  {/* Left Callout */}
                  <div className="font-mono text-xs space-y-1 z-10">
                    <div className="text-[10px] text-slate-500 uppercase">COGNITIVE RESOLVER</div>
                    <div className="text-argon font-bold flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-argon animate-ping"></span>
                      STAGE 0/1 READY
                    </div>
                    <div className="text-[11px] text-slate-400">Zero-LLM: &lt;5ms • 85% Thresh</div>
                  </div>

                  {/* Center Vector Radar */}
                  <div className="relative w-20 h-20 flex items-center justify-center">
                    <svg className="w-full h-full" viewBox="0 0 100 100">
                      <circle cx="50" cy="50" r="45" fill="none" stroke="#1A293E" strokeWidth="1" strokeDasharray="3 3" />
                      <circle cx="50" cy="50" r="35" fill="none" stroke="#1A293E" strokeWidth="1" />
                      <circle cx="50" cy="50" r="22" fill="none" stroke="var(--primary)" strokeWidth="1.5" opacity="0.6" />
                      <line x1="50" y1="50" x2="50" y2="5" stroke="var(--primary)" strokeWidth="1.5" className="radar-beam" opacity="0.8" />
                      <circle cx="50" cy="50" r="4" fill="var(--primary)" />
                    </svg>
                    <div className="absolute text-[8px] font-mono font-bold tracking-tighter text-argon">
                      {hubStatus.toUpperCase()}
                    </div>
                  </div>

                  {/* Right Callout */}
                  <div className="font-mono text-xs space-y-1 text-right z-10 flex flex-col items-end">
                    <div className="text-[10px] text-slate-500 uppercase">LAST DISPATCH</div>
                    <div className="text-slate-200 font-bold">{lastReply.provider}</div>
                    <div className="text-[11px] text-emerald-400">{lastReply.latency}ms • {lastReply.src}</div>
                  </div>
                </div>
              )}

              {/* View Selector Tabs */}
              <div className="flex items-center gap-1 bg-surface border border-border p-1 rounded font-mono text-xs">
                {['Chat', 'Tasks', 'Memory', 'Projects'].map(view => (
                  <button
                    key={view}
                    type="button"
                    onClick={() => setActiveView(view)}
                    className={`flex-1 py-1 rounded transition-colors ${activeView === view ? 'bg-argon/20 text-argon font-bold border border-argon/40' : 'text-slate-400 hover:text-slate-200'}`}
                  >
                    {view.toUpperCase()}
                  </button>
                ))}
              </div>

              {/* Console Body */}
              <div className="glass-bracket flex-1 p-3 rounded flex flex-col overflow-hidden">
                
                {/* 1. CHAT VIEW */}
                {activeView === 'Chat' && (
                  <div className="flex-1 flex flex-col overflow-hidden">
                    
                    {/* Message Stream */}
                    <div className="flex-1 overflow-y-auto space-y-3.5 pr-1 text-xs">
                      {messages.map((m, idx) => {
                        const priorUserPrompt = m.role === 'assistant' && idx > 0 ? messages[idx - 1].text : '';
                        const isCopied = copiedId === m.id;
                        return (
                          <div key={m.id} className={`group flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
                            
                            {/* Message Bubble - 100% Selectable */}
                            <div className={`p-3 rounded-lg max-w-[92%] leading-relaxed shadow-sm transition-all chat-selectable ${
                              m.role === 'user' 
                                ? 'bg-argon/15 text-slate-100 border border-argon/30' 
                                : 'bg-[#0A1017] text-slate-200 border border-border'
                            }`}>
                              {m.role === 'user' ? (
                                <div className="font-mono text-xs whitespace-pre-wrap chat-selectable">{m.text}</div>
                              ) : (
                                <FormattedMessage 
                                  text={m.text} 
                                  onOpenArtifact={(art) => {
                                    setCurrentArtifact(art);
                                    setArtifactTab('preview');
                                  }} 
                                />
                              )}
                            </div>

                            {/* Visible Action Bar on EVERY chat bubble */}
                            <div className="flex items-center gap-2 mt-1 text-[10px] font-mono text-slate-400 px-1 opacity-90 group-hover:opacity-100 transition-opacity select-none">
                              {m.role === 'assistant' && (
                                <>
                                  <span className="text-argon">[{m.provider}]</span>
                                  <span>•</span>
                                  <span>{m.src}</span>
                                  <span>•</span>
                                  <span>{m.lat}ms</span>
                                  <span>|</span>
                                </>
                              )}

                              {/* Action buttons */}
                              <button
                                type="button"
                                onClick={() => handleCopyMessage(m.id, m.text)}
                                className={`hover:text-argon transition-colors flex items-center gap-1 font-bold ${isCopied ? 'text-emerald-400' : ''}`}
                                title="Copy Message Text"
                              >
                                {isCopied ? '✓ COPIED!' : '📋 COPY MESSAGE'}
                              </button>

                              {m.role === 'assistant' && (
                                <>
                                  <span>|</span>
                                  <button
                                    type="button"
                                    onClick={() => handleSendChat(priorUserPrompt, 'gemini')}
                                    className="hover:text-blue-400 text-blue-300 transition-colors flex items-center gap-0.5"
                                    title="Retry with Gemini Deep Pro"
                                  >
                                    <span>🔵</span> GEMINI
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => handleSendChat(priorUserPrompt, 'groq')}
                                    className="hover:text-amber-400 text-amber-300 transition-colors flex items-center gap-0.5"
                                    title="Retry with Groq LPU"
                                  >
                                    <span>🟠</span> GROQ
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => handleSendChat(priorUserPrompt, 'web_search')}
                                    className="hover:text-purple-400 text-purple-300 transition-colors flex items-center gap-0.5"
                                    title="Retry with Web Search Grounding"
                                  >
                                    <span>🌐</span> WEB
                                  </button>
                                  <span>|</span>
                                  <button
                                    type="button"
                                    onClick={() => handleApprovePair(m.id, priorUserPrompt, m.text)}
                                    className={`transition-colors font-bold ${approvedPairs[m.id] ? 'text-emerald-400' : 'text-slate-400 hover:text-amber-400'}`}
                                    title="Approve pair for Golden Fine-Tune Dataset"
                                  >
                                    {approvedPairs[m.id] ? '✓ APPROVED' : '👍 FINE-TUNE'}
                                  </button>
                                </>
                              )}

                              {m.role === 'user' && (
                                <>
                                  <span>|</span>
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setInputVal(m.text);
                                      textareaRef.current?.focus();
                                    }}
                                    className="hover:text-argon transition-colors"
                                    title="Edit prompt in input box"
                                  >
                                    ✏️ EDIT
                                  </button>
                                </>
                              )}
                            </div>

                          </div>
                        );
                      })}
                      <div ref={chatEndRef} />
                    </div>

                    {/* Quick Command Pills */}
                    <div className="flex items-center gap-1.5 py-1.5 overflow-x-auto text-[10px] font-mono text-slate-400 border-t border-border/50">
                      <button 
                        type="button"
                        onClick={() => handleSendChat('who am I')} 
                        className="px-2 py-0.5 rounded bg-slate-900 border border-border hover:border-argon whitespace-nowrap"
                      >
                        ⚡ who am I
                      </button>
                      <button 
                        type="button"
                        onClick={() => handleSendChat('full profile')} 
                        className="px-2 py-0.5 rounded bg-slate-900 border border-border hover:border-argon whitespace-nowrap"
                      >
                        📋 full profile
                      </button>
                      <button 
                        type="button"
                        onClick={() => handleSendChat('what tasks are left?')} 
                        className="px-2 py-0.5 rounded bg-slate-900 border border-border hover:border-argon whitespace-nowrap"
                      >
                        ⚡ what tasks are left?
                      </button>
                      <button 
                        type="button"
                        onClick={() => handleSendChat('dataset stats')} 
                        className="px-2 py-0.5 rounded bg-slate-900 border border-border hover:border-argon whitespace-nowrap"
                      >
                        📊 dataset stats
                      </button>
                      <button 
                        type="button"
                        onClick={() => handleSendChat('kernel status')} 
                        className="px-2 py-0.5 rounded bg-slate-900 border border-border hover:border-argon whitespace-nowrap"
                      >
                        ⚙️ kernel status
                      </button>
                    </div>

                    {/* 4-Brain Target Selector Bar Above Composer */}
                    <div className="flex items-center gap-1 py-1.5 overflow-x-auto text-[10px] font-mono border-t border-border/40 select-none">
                      <span className="text-slate-500 mr-1 text-[9px] uppercase">TARGET BRAIN:</span>
                      
                      <button
                        type="button"
                        onClick={() => setSelectedBrain('auto')}
                        className={`px-2 py-0.5 rounded border transition-colors whitespace-nowrap font-bold ${
                          selectedBrain === 'auto' ? 'bg-argon/20 text-argon border-argon shadow-[0_0_8px_rgba(86,226,245,0.3)]' : 'border-border text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        ⚡ AUTO (Arbiter)
                      </button>

                      <button
                        type="button"
                        onClick={() => setSelectedBrain('kernel')}
                        className={`px-2 py-0.5 rounded border transition-colors whitespace-nowrap font-bold ${
                          selectedBrain === 'kernel' ? 'bg-cyan-500/20 text-cyan-300 border-cyan-400' : 'border-border text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        🧠 KERNEL (0-LLM)
                      </button>

                      <button
                        type="button"
                        onClick={() => setSelectedBrain('groq')}
                        className={`px-2 py-0.5 rounded border transition-colors whitespace-nowrap font-bold ${
                          selectedBrain === 'groq' ? 'bg-amber-500/20 text-amber-300 border-amber-400' : 'border-border text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        🟠 GROQ (Llama 70B)
                      </button>

                      <button
                        type="button"
                        onClick={() => setSelectedBrain('gemini')}
                        className={`px-2 py-0.5 rounded border transition-colors whitespace-nowrap font-bold ${
                          selectedBrain === 'gemini' ? 'bg-blue-500/20 text-blue-300 border-blue-400' : 'border-border text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        🔵 GEMINI (Deep Pro)
                      </button>

                      <button
                        type="button"
                        onClick={() => setSelectedBrain('web_search')}
                        className={`px-2 py-0.5 rounded border transition-colors whitespace-nowrap font-bold ${
                          selectedBrain === 'web_search' ? 'bg-purple-500/20 text-purple-300 border-purple-400' : 'border-border text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        🌐 WEB SEARCH
                      </button>

                      {selectedBrain !== 'auto' && (
                        <button
                          type="button"
                          onClick={() => setSelectedBrain('auto')}
                          className="px-2 py-0.5 rounded border border-rose-500/50 bg-rose-500/15 text-rose-300 hover:bg-rose-500/25 transition-colors whitespace-nowrap font-bold"
                        >
                          ✕ RESET TO AUTO
                        </button>
                      )}

                      <button
                        type="button"
                        onClick={() => setSelectedBrain('claude')}
                        className={`px-2 py-0.5 rounded border transition-colors whitespace-nowrap font-bold ${
                          selectedBrain === 'claude' ? 'bg-fuchsia-500/20 text-fuchsia-300 border-fuchsia-400' : 'border-border text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        🟣 CLAUDE (In-Chat)
                      </button>

                      <button
                        type="button"
                        onClick={() => setSelectedBrain('chatgpt')}
                        className={`px-2 py-0.5 rounded border transition-colors whitespace-nowrap font-bold ${
                          selectedBrain === 'chatgpt' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-400' : 'border-border text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        🟢 CHATGPT (In-Chat)
                      </button>

                      <button
                        type="button"
                        onClick={() => setSelectedBrain('grok')}
                        className={`px-2 py-0.5 rounded border transition-colors whitespace-nowrap font-bold ${
                          selectedBrain === 'grok' ? 'bg-slate-300/20 text-slate-100 border-slate-300' : 'border-border text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        ⚪ GROK (In-Chat)
                      </button>
                    </div>

                    {/* Attachment Chips Bar (Smart Auto-Paste & Files) */}
                    {attachments.length > 0 && (
                      <div className="flex items-center gap-1.5 pb-2 overflow-x-auto">
                        {attachments.map(att => (
                          <div key={att.id} className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-argon/15 border border-argon/40 text-argon text-[10px] font-mono">
                            <span>📄 {att.name} ({att.size} KB / {att.lines} lines)</span>
                            <button 
                              type="button"
                              onClick={() => setAttachments(prev => prev.filter(a => a.id !== att.id))} 
                              className="text-slate-400 hover:text-rose-400"
                            >
                              ✕
                            </button>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Auto-Expanding Chat Composer Bar */}
                    <form 
                      onSubmit={(e) => { e.preventDefault(); handleSendChat(); }} 
                      className="flex items-end gap-2 bg-[#060A10] border border-border rounded-lg p-1.5 focus-within:border-argon transition-colors"
                    >
                      <input 
                        type="file" 
                        ref={fileInputRef} 
                        onChange={handleFileUpload} 
                        className="hidden" 
                        accept=".txt,.py,.js,.jsx,.ts,.tsx,.json,.html,.css,.md,.csv,.sh,.yaml,.yml,.sql" 
                      />

                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="p-1.5 text-slate-400 hover:text-argon transition-colors text-sm"
                        title="Attach local code/text file (📎)"
                      >
                        📎
                      </button>

                      <textarea
                        ref={textareaRef}
                        rows={1}
                        value={inputVal}
                        onChange={(e) => {
                          setInputVal(e.target.value);
                          e.target.style.height = 'auto';
                          e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`;
                        }}
                        onPaste={handlePaste}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSendChat();
                          }
                        }}
                        placeholder="Message AIVCO... (Enter to send, Shift+Enter for newline)"
                        className="flex-1 bg-transparent border-0 text-xs font-mono text-slate-200 outline-none resize-none py-1.5 px-1 max-h-40 leading-relaxed chat-selectable"
                        disabled={isSending}
                      />

                      {isSending ? (
                        <button
                          type="button"
                          onClick={() => {
                            setIsSending(false);
                            setHubStatus('idle');
                          }}
                          className="px-3 py-1.5 bg-rose-500/20 text-rose-400 border border-rose-500/40 font-mono text-xs font-bold rounded hover:bg-rose-500/30 flex items-center gap-1"
                        >
                          <span>⏹</span> STOP
                        </button>
                      ) : (
                        <button 
                          type="submit" 
                          disabled={!inputVal.trim() && attachments.length === 0}
                          className="px-4 py-1.5 bg-argon text-obsidian font-mono text-xs font-bold rounded hover:opacity-90 disabled:opacity-40 transition-opacity"
                        >
                          TRANSMIT
                        </button>
                      )}
                    </form>

                  </div>
                )}

                {/* 2. TASKS VIEW */}
                {activeView === 'Tasks' && (
                  <div className="flex-1 flex flex-col overflow-hidden">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-xs text-slate-400">ACTIVE TASK LIST [{state.tasks.length}]</span>
                    </div>
                    <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
                      {state.tasks.map((task) => (
                        <div 
                          key={task.id}
                          onClick={() => handleToggleTask(task)}
                          className={`p-2 rounded border flex items-center justify-between cursor-pointer text-xs font-mono transition-colors ${
                            task.done ? 'border-border/40 bg-slate-950/20 text-slate-500 line-through' : 'border-border bg-card text-slate-200 hover:border-argon'
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <span className={task.done ? 'text-emerald-500' : 'text-slate-600'}>
                              {task.done ? '✓' : '○'}
                            </span>
                            <span>{task.text}</span>
                          </div>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded border ${
                            task.tag === 'Priority' ? 'border-rose-500/40 text-rose-400 bg-rose-500/10' : 'border-border text-slate-400'
                          }`}>
                            {task.tag}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 3. MEMORY VIEW */}
                {activeView === 'Memory' && (
                  <div className="flex-1 overflow-y-auto space-y-3 text-xs pr-1">
                    <div className="font-mono text-slate-400 text-xs flex justify-between">
                      <span>ENGRAM REPOSITORY</span>
                      <span className="text-argon">STAGE 1 GROUNDED</span>
                    </div>
                    {Object.entries(state.layers || {}).map(([layer, engrams]) => (
                      <div key={layer} className="border border-border rounded p-2.5 bg-surface/40">
                        <div className="font-mono font-bold text-argon text-xs mb-1.5 flex justify-between">
                          <span>{layer.toUpperCase()} LAYER</span>
                          <span className="text-slate-500">[{engrams.length}]</span>
                        </div>
                        <div className="space-y-1.5">
                          {engrams.map((e, idx) => (
                            <div key={idx} className="p-1.5 rounded bg-slate-900/60 border border-border/50 text-[11px]">
                              <div className="font-bold text-slate-300">{e.title}</div>
                              <div className="text-slate-400 text-[10px] mt-0.5">{e.body}</div>
                              <div className="text-slate-600 text-[9px] font-mono mt-1">{e.time} • {e.tag}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* 4. PROJECTS VIEW */}
                {activeView === 'Projects' && (
                  <div className="flex-1 overflow-y-auto space-y-3 text-xs pr-1">
                    <div className="font-mono text-slate-400 text-xs flex justify-between">
                      <span>PROJECT REGISTRY</span>
                      <button type="button" onClick={handleCreateProject} className="text-argon">+ CREATE PROJECT</button>
                    </div>
                    {Object.entries(state.projects || {}).map(([pid, milestones]) => (
                      <div key={pid} className={`border rounded p-3 ${pid === state.active_project ? 'border-argon bg-argon/5' : 'border-border bg-card'}`}>
                        <div className="flex justify-between font-mono text-xs mb-2">
                          <span className="font-bold text-slate-100">{pid.toUpperCase()}</span>
                          {pid === state.active_project && <span className="text-argon font-bold">[ACTIVE]</span>}
                        </div>
                        <div className="space-y-1.5">
                          {milestones.map(([m, pct], i) => (
                            <div key={i} className="text-[11px] flex justify-between text-slate-300 font-mono">
                              <span>• {m}</span>
                              <span className="text-argon">{pct}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

              </div>
            </main>

            {/* ── RIGHT TELEMETRY & TASK SIDEBAR (Col 10-12) ── */}
            <aside className="col-span-3 flex flex-col gap-2 overflow-y-auto pl-1">
              
              {/* Telemetry & Fine-Tune Pair Counter */}
              <div className="glass-bracket p-3 rounded flex flex-col gap-2 font-mono text-xs">
                <div className="text-slate-400 uppercase text-[10px]">HARDWARE & KERNEL TELEMETRY</div>
                
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-[11px] text-slate-300 mb-0.5">
                      <span>RAM ALLOCATION (8GB)</span>
                      <span className="text-argon">3.4 GB</span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-argon w-[42%]"></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] text-slate-300 mb-0.5">
                      <span>CONTEXT UTILIZATION</span>
                      <span className="text-argon">18%</span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-400 w-[18%]"></div>
                    </div>
                  </div>
                </div>

                {/* LoRA Dataset Logger Counter */}
                <div className="mt-2 p-2 rounded bg-slate-900 border border-border">
                  <div className="text-[10px] text-slate-400">FINE-TUNE DATASET (ShareGPT)</div>
                  <div className="text-base font-bold text-argon">
                    {state.dataset_stats?.total_entries || 0} PAIRS LOGGED
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">Ready for Mac LoRA / MLX Export</div>
                </div>
              </div>

              {/* 3D HTML5 Canvas Constellation Sphere */}
              <ConstellationSphere themeColor={THEME_COLORS[theme]} />

              {/* Active Tasks Interactive Checklist */}
              <div className="glass-bracket p-3 rounded flex-1 flex flex-col gap-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-400 uppercase">ACTIVE TASKS</span>
                  <span className="text-argon font-bold">{state.tasks.filter(t => !t.done).length} PENDING</span>
                </div>

                <form onSubmit={handleAddTask} className="flex gap-1.5">
                  <input 
                    type="text" 
                    value={newTaskText} 
                    onChange={(e) => setNewTaskText(e.target.value)}
                    placeholder="+ Add new task..."
                    className="flex-1 bg-obsidian border border-border rounded px-2 py-1 text-xs font-mono text-slate-200 outline-none focus:border-argon"
                  />
                  <button type="submit" className="px-2 py-1 bg-argon text-obsidian text-xs font-bold rounded">
                    +
                  </button>
                </form>

                <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
                  {state.tasks.map((task) => (
                    <div 
                      key={task.id}
                      onClick={() => handleToggleTask(task)}
                      className={`p-2 rounded border flex items-center gap-2 cursor-pointer text-xs font-mono transition-colors ${
                        task.done ? 'border-border/40 bg-slate-950/20 text-slate-500 line-through' : 'border-border bg-card text-slate-200 hover:border-argon'
                      }`}
                    >
                      <input 
                        type="checkbox" 
                        checked={task.done} 
                        readOnly 
                        className="accent-cyan-400 cursor-pointer pointer-events-none" 
                      />
                      <span className="truncate">{task.text}</span>
                    </div>
                  ))}
                  {state.tasks.length === 0 && (
                    <div className="text-xs text-slate-600 italic">No tasks in active project</div>
                  )}
                </div>
              </div>

            </aside>

            {/* ── CLAUDE-STYLE ARTIFACT SLIDE-OVER DRAWER ── */}
            {currentArtifact && (
              <div className={`fixed top-12 bottom-0 right-0 z-40 bg-[#070B12] border-l border-border shadow-2xl flex flex-col transition-all duration-300 ${
                isArtifactFullscreen ? 'w-full' : 'w-full md:w-[620px] lg:w-[740px]'
              }`}>
                {/* Drawer Top Header */}
                <div className="h-12 bg-[#0E1724] border-b border-border px-4 flex items-center justify-between text-xs font-mono select-none">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-argon animate-pulse"></span>
                    <span className="font-bold text-slate-100">{currentArtifact.title}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-argon/20 text-argon border border-argon/30 uppercase">
                      {currentArtifact.lang}
                    </span>
                  </div>

                  {/* Tabs: LIVE PREVIEW vs CODE */}
                  <div className="flex items-center gap-1 bg-surface border border-border p-0.5 rounded">
                    <button
                      type="button"
                      onClick={() => setArtifactTab('preview')}
                      className={`px-3 py-1 rounded transition-colors ${artifactTab === 'preview' ? 'bg-argon text-obsidian font-bold' : 'text-slate-400 hover:text-white'}`}
                    >
                      ▶ LIVE PREVIEW
                    </button>
                    <button
                      type="button"
                      onClick={() => setArtifactTab('code')}
                      className={`px-3 py-1 rounded transition-colors ${artifactTab === 'code' ? 'bg-argon text-obsidian font-bold' : 'text-slate-400 hover:text-white'}`}
                    >
                      &lt;/&gt; CODE
                    </button>
                  </div>

                  {/* Controls */}
                  <div className="flex items-center gap-2">
                    {artifactTab === 'preview' && (
                      <button
                        type="button"
                        onClick={() => setIframeKey(k => k + 1)}
                        className="px-2 py-1 rounded border border-border text-slate-300 hover:border-argon hover:text-argon transition-colors font-bold"
                        title="Refresh Preview (↺)"
                      >
                        ↺
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => copyToClipboard(currentArtifact.code)}
                      className="px-2 py-1 rounded border border-border text-slate-300 hover:border-argon hover:text-argon transition-colors"
                      title="Copy code"
                    >
                      COPY
                    </button>
                    <button
                      type="button"
                      onClick={() => setIsArtifactFullscreen(!isArtifactFullscreen)}
                      className="px-2 py-1 rounded border border-border text-slate-300 hover:border-argon hover:text-argon transition-colors"
                      title="Toggle Fullscreen (⤢)"
                    >
                      {isArtifactFullscreen ? 'COLLAPSE ⤢' : 'EXPAND ⤢'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setCurrentArtifact(null)}
                      className="w-7 h-7 rounded border border-border text-slate-400 hover:text-white hover:border-rose-500 hover:bg-rose-500/20 flex items-center justify-center font-bold"
                      title="Close Drawer"
                    >
                      ✕
                    </button>
                  </div>
                </div>

                {/* Drawer Body */}
                <div className="flex-1 overflow-hidden relative bg-[#04070B]">
                  {artifactTab === 'preview' ? (
                    <iframe
                      key={iframeKey}
                      title="Artifact Sandbox Live Preview"
                      sandbox="allow-scripts allow-modals allow-same-origin allow-forms"
                      srcDoc={currentArtifact.code}
                      className="w-full h-full bg-white border-0"
                    />
                  ) : (
                    <div className="h-full overflow-auto p-4 text-xs font-mono">
                      <pre className="text-slate-200 leading-relaxed chat-selectable">
                        <code>{currentArtifact.code}</code>
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            )}

          </div>

          {/* ── MODALS ── */}

          {/* New Engram Modal */}
          {showNewEngramModal && (
            <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
              <div className="glass-bracket p-5 rounded max-w-md w-full bg-card border border-border flex flex-col gap-3 font-mono text-xs">
                <div className="flex justify-between items-center text-argon font-bold text-sm">
                  <span>+ COMMIT NEW ENGRAM</span>
                  <button type="button" onClick={() => setShowNewEngramModal(false)} className="text-slate-400 hover:text-white">✕</button>
                </div>
                
                <label className="text-slate-400">Target Memory Layer</label>
                <select 
                  value={newEngram.layer} 
                  onChange={(e) => setNewEngram({ ...newEngram, layer: e.target.value })}
                  className="bg-obsidian border border-border rounded p-2 text-slate-200 outline-none focus:border-argon"
                >
                  <option value="Semantic">Semantic (Facts & Concepts)</option>
                  <option value="Procedural">Procedural (Rules & Workflows)</option>
                  <option value="Episodic">Episodic (Events & Milestones)</option>
                  <option value="Creative">Creative (Ideas & Architecture)</option>
                </select>

                <label className="text-slate-400">Engram Title / Trigger</label>
                <input 
                  type="text" 
                  value={newEngram.title} 
                  onChange={(e) => setNewEngram({ ...newEngram, title: e.target.value })}
                  placeholder="e.g. favorite_ide or rule trigger"
                  className="bg-obsidian border border-border rounded p-2 text-slate-200 outline-none focus:border-argon"
                />

                <label className="text-slate-400">Engram Content</label>
                <textarea 
                  rows="4"
                  value={newEngram.body} 
                  onChange={(e) => setNewEngram({ ...newEngram, body: e.target.value })}
                  placeholder="Detailed fact, rule logic, or episodic summary..."
                  className="bg-obsidian border border-border rounded p-2 text-slate-200 outline-none focus:border-argon"
                />

                <div className="flex justify-end gap-2 mt-2">
                  <button 
                    type="button"
                    onClick={() => setShowNewEngramModal(false)}
                    className="px-3 py-1.5 rounded border border-border text-slate-400 hover:text-white"
                  >
                    Cancel
                  </button>
                  <button 
                    type="button"
                    onClick={handleAddEngram}
                    className="px-4 py-1.5 rounded bg-argon text-obsidian font-bold"
                  >
                    Save Engram
                  </button>
                </div>
              </div>
            </div>
          )}


          {/* AIVCO Live Browser Bridge Modal */}
          {showVaultModal && (
            <div className="fixed inset-0 bg-black/85 backdrop-blur-md z-50 flex items-center justify-center p-4">
              <div className="glass-bracket p-6 rounded-xl max-w-3xl w-full bg-[#0A101A] border border-argon/50 flex flex-col gap-5 font-mono text-xs shadow-[0_0_40px_rgba(86,226,245,0.2)]">
                {/* Header */}
                <div className="flex justify-between items-center border-b border-border/80 pb-3">
                  <div className="flex items-center gap-2.5">
                    <span className="text-xl">🌐</span>
                    <div>
                      <div className="font-bold text-sm tracking-wider text-argon">AIVCO LIVE BROWSER BRIDGE (PORT 9222 CDP)</div>
                      <div className="text-[11px] text-slate-400 font-sans">
                        Zero Profile Locks • Persistent Chromium Session • Direct DOM Streaming
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => { checkCdpStatus(); fetchVaultStatus(); }}
                      disabled={vaultLoading}
                      className="px-2.5 py-1 rounded border border-argon/40 text-argon bg-argon/10 hover:bg-argon/25 flex items-center gap-1 text-[11px]"
                    >
                      <span>🔄</span>
                      <span>{vaultLoading ? 'Checking...' : 'Refresh Status'}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowVaultModal(false)}
                      className="text-slate-400 hover:text-white px-2 py-1 text-sm font-bold"
                    >
                      ✕
                    </button>
                  </div>
                </div>

                {/* Primary Action Hero */}
                <div className="p-4 rounded-xl border border-argon/40 bg-gradient-to-r from-argon/10 via-surface/60 to-surface/80 flex flex-col sm:flex-row justify-between items-center gap-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-sm text-slate-100">Live Browser Session</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${cdpStatus?.running ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40' : 'bg-amber-500/20 text-amber-300 border-amber-500/40'}`}>
                        {cdpStatus?.running ? '🟢 Connected (Port 9222)' : '⚪ Offline / Not Running'}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 mt-1 font-sans">
                      Opens Chrome with isolated profile <code className="text-argon">data/browser_profiles/primary_vault</code> and tabs for all 4 AI platforms.
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={handleLaunchCdp}
                    className="px-5 py-2.5 rounded-lg bg-argon text-obsidian font-bold text-xs hover:bg-argon/90 active:scale-95 transition-all shadow-[0_0_20px_rgba(86,226,245,0.4)] whitespace-nowrap flex items-center gap-2"
                  >
                    <span>🌐</span>
                    <span>Launch / Focus Agent Browser</span>
                  </button>
                </div>

                {/* 4 Core Web Platforms */}
                <div>
                  <div className="text-[11px] font-bold text-slate-300 uppercase tracking-wider mb-2.5 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-argon"></span>
                    <span>Connected AI Web Platforms</span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {[
                      { key: 'claude', name: 'Claude 3.7 Sonnet', url: 'https://claude.ai', domain: 'claude.ai', color: '#d97706' },
                      { key: 'chatgpt', name: 'ChatGPT 4o', url: 'https://chatgpt.com', domain: 'chatgpt.com', color: '#10a37f' },
                      { key: 'gemini', name: 'Gemini Pro Web', url: 'https://gemini.google.com/app', domain: 'gemini.google.com', color: '#4285f4' },
                      { key: 'grok', name: 'Grok 3 Web', url: 'https://grok.com', domain: 'grok.com', color: '#ff4655' }
                    ].map(prov => {
                      const isConnected = cdpStatus?.running;
                      return (
                        <div key={prov.key} className="p-3.5 rounded-lg bg-surface/80 border border-border flex flex-col justify-between gap-3">
                          <div className="flex justify-between items-start">
                            <div>
                              <div className="font-bold text-slate-100 flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: prov.color }}></span>
                                <span>{prov.name}</span>
                              </div>
                              <div className="text-[10px] text-slate-400 mt-0.5">{prov.domain}</div>
                            </div>
                            <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${
                              isConnected ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40' : 'bg-slate-700/30 text-slate-400 border-slate-600/40'
                            }`}>
                              {isConnected ? '🟢 Live Tab Ready' : '⚪ Launch Browser First'}
                            </span>
                          </div>
                          <div className="text-[10px] text-slate-400 font-sans">
                            {isConnected ? 'Session active in open browser. AIVCO connects live on dispatch.' : 'Launch browser, sign in once, and keep window open.'}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Footer Notice & Instructions */}
                <div className="border-t border-border/80 pt-3 flex flex-col sm:flex-row justify-between items-center gap-3 text-[11px] text-slate-400 font-sans">
                  <div className="flex items-center gap-1.5">
                    <span>💡</span>
                    <span><b>How it works:</b> Click Launch Browser, log in once, and <b>leave the Chrome window open</b>. AIVCO automates it in real time!</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowVaultModal(false)}
                    className="px-4 py-1.5 rounded bg-argon text-obsidian font-bold hover:bg-argon/90 font-mono text-xs"
                  >
                    Done
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Global Search Modal (Ctrl+K) */}
          {showSearchModal && (
            <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
              <div className="glass-bracket p-4 rounded max-w-lg w-full bg-card border border-border flex flex-col gap-3 font-mono text-xs">
                <div className="flex justify-between items-center text-argon font-bold">
                  <span>GLOBAL ENGRAM SEARCH</span>
                  <button type="button" onClick={() => setShowSearchModal(false)} className="text-slate-400 hover:text-white">✕</button>
                </div>
                <input 
                  type="text" 
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Type to search engrams, tasks, or commands..."
                  className="bg-obsidian border border-border rounded p-2 text-slate-200 outline-none focus:border-argon"
                  autoFocus
                />
                <div className="max-h-60 overflow-y-auto space-y-1.5">
                  {searchQuery.trim() && Object.values(state.layers || {}).flat()
                    .filter(item => 
                      (item.title || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
                      (item.body || '').toLowerCase().includes(searchQuery.toLowerCase())
                    )
                    .map((item, idx) => (
                      <div 
                        key={idx} 
                        className="p-2 rounded bg-slate-900 border border-border/60 hover:border-argon cursor-pointer"
                        onClick={() => {
                          handleSendChat(`recall ${item.title}`);
                          setShowSearchModal(false);
                        }}
                      >
                        <div className="font-bold text-slate-200">{item.title}</div>
                        <div className="text-slate-400 text-[10px] truncate">{item.body}</div>
                      </div>
                    ))}
                  {searchQuery.trim() && (
                    <div 
                      className="p-2 rounded bg-argon/10 border border-argon/30 hover:bg-argon/20 cursor-pointer text-argon"
                      onClick={() => {
                        handleSendChat(searchQuery);
                        setShowSearchModal(false);
                      }}
                    >
                      Ask AIVCO: &quot;{searchQuery}&quot;
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById('root')).render(<App />);
  