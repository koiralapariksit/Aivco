with open("index.html", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 1. Update State variables (lines 615-632)
# Find index where `// Live Browser Bridge (CDP) State` starts
idx_state = -1
for i, line in enumerate(lines):
    if "// Live Browser Bridge (CDP) State" in line:
        idx_state = i
        break

if idx_state != -1:
    # Find where `const handleLaunchCdp` starts
    idx_state_end = -1
    for i in range(idx_state, idx_state + 30):
        if "const handleLaunchCdp" in lines[i]:
            idx_state_end = i
            break
    
    new_state_lines = [
        "      // Native Chrome Extension Bridge (Manifest V3) State\n",
        "      const [extensionStatus, setExtensionStatus] = useState({ connected: false, tabs: {}, client_count: 0 });\n",
        "\n",
        "      const checkExtensionStatus = useCallback(async () => {\n",
        "        try {\n",
        "          const res = await fetch(`${API_BASE}/api/browser/extension_status`);\n",
        "          if (res.ok) {\n",
        "            const data = await res.json();\n",
        "            setExtensionStatus(data);\n",
        "          }\n",
        "        } catch (_) {}\n",
        "      }, []);\n",
        "\n",
        "      useEffect(() => {\n",
        "        checkExtensionStatus();\n",
        "        const iv = setInterval(checkExtensionStatus, 3000);\n",
        "        return () => clearInterval(iv);\n",
        "      }, [checkExtensionStatus]);\n",
        "\n",
        "      // Backward compatibility stubs\n",
        "      const cdpStatus = { running: extensionStatus.connected };\n",
        "      const checkCdpStatus = checkExtensionStatus;\n",
        "\n",
    ]
    lines[idx_state:idx_state_end] = new_state_lines
    print(f"Updated state lines from index {idx_state} to {idx_state_end}")

# 2. Update Top Bar Button
idx_btn = -1
for i, line in enumerate(lines):
    if 'id="connectWebAccountsBtn"' in line:
        idx_btn = i - 1  # the <button line
        break

if idx_btn != -1:
    idx_btn_end = -1
    for i in range(idx_btn, idx_btn + 30):
        if "</button>" in lines[i]:
            idx_btn_end = i + 1
            break
    
    new_btn_lines = [
        "              <button \n",
        '                id="extensionBridgeBtn"\n',
        '                type="button"\n',
        "                onClick={() => {\n",
        "                  setShowVaultModal(true);\n",
        "                  checkExtensionStatus();\n",
        "                }}\n",
        "                className={`px-3 py-1 rounded border font-mono text-xs transition-all flex items-center gap-1.5 font-bold cursor-pointer select-none ${\n",
        "                  extensionStatus?.connected\n",
        "                    ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50 shadow-[0_0_12px_rgba(16,185,129,0.3)]'\n",
        "                    : 'bg-red-500/20 text-red-400 border-red-500/50 shadow-[0_0_12px_rgba(239,68,68,0.25)]'\n",
        "                }`}\n",
        "                style={{ cursor: 'pointer', pointerEvents: 'auto', zIndex: 1000 }}\n",
        '                title="AIVCO Native Chrome Extension Bridge (Manifest V3)"\n',
        "              >\n",
        "                {extensionStatus?.connected ? (\n",
        "                  <>\n",
        "                    <span>🟢</span>\n",
        "                    <span>EXTENSION: CONNECTED</span>\n",
        '                    <span className="text-[10px] opacity-80 font-normal">\n',
        "                      ({Object.keys(extensionStatus.tabs || {}).filter(k => extensionStatus.tabs[k]).map(k => k.charAt(0).toUpperCase() + k.slice(1)).join(', ') || 'Claude, ChatGPT, Gemini, Grok'})\n",
        "                    </span>\n",
        "                  </>\n",
        "                ) : (\n",
        "                  <>\n",
        "                    <span>🔌</span>\n",
        "                    <span>EXTENSION: DISCONNECTED</span>\n",
        "                  </>\n",
        "                )}\n",
        "              </button>\n",
    ]
    lines[idx_btn:idx_btn_end] = new_btn_lines
    print(f"Updated top bar button from index {idx_btn} to {idx_btn_end}")

# 3. Update Modal
idx_modal = -1
for i, line in enumerate(lines):
    if "{/* AIVCO Live Browser Bridge Modal */}" in line:
        idx_modal = i
        break

if idx_modal != -1:
    idx_modal_end = -1
    for i in range(idx_modal, len(lines)):
        if "{/* Global Search Modal (Ctrl+K) */}" in lines[i]:
            idx_modal_end = i
            break
    
    new_modal_lines = [
        "          {/* AIVCO Native Chrome Extension Bridge Modal */}\n",
        "          {showVaultModal && (\n",
        '            <div className="fixed inset-0 bg-black/85 backdrop-blur-md z-50 flex items-center justify-center p-4">\n',
        '              <div className="glass-bracket p-6 rounded-xl max-w-2xl w-full bg-[#0A101A] border border-argon/50 flex flex-col gap-5 font-mono text-xs shadow-[0_0_40px_rgba(86,226,245,0.2)]">\n',
        "                {/* Header */}\n",
        '                <div className="flex justify-between items-center border-b border-border/80 pb-3">\n',
        '                  <div className="flex items-center gap-2.5">\n',
        '                    <span className="text-xl">⚡</span>\n',
        "                    <div>\n",
        '                      <div className="font-bold text-sm tracking-wider text-argon">AIVCO NATIVE CHROME EXTENSION BRIDGE</div>\n',
        '                      <div className="text-[11px] text-slate-400 font-sans">\n',
        "                        Manifest V3 • Authentic Chrome Session • Zero Bot Detection\n",
        "                      </div>\n",
        "                    </div>\n",
        "                  </div>\n",
        '                  <div className="flex items-center gap-2">\n',
        "                    <button\n",
        '                      type="button"\n',
        "                      onClick={checkExtensionStatus}\n",
        '                      className="px-2.5 py-1 rounded border border-argon/40 text-argon bg-argon/10 hover:bg-argon/25 flex items-center gap-1 text-[11px] cursor-pointer"\n',
        "                    >\n",
        "                      <span>🔄</span>\n",
        "                      <span>Refresh Status</span>\n",
        "                    </button>\n",
        "                    <button\n",
        '                      type="button"\n',
        "                      onClick={() => setShowVaultModal(false)}\n",
        '                      className="text-slate-400 hover:text-white px-2 py-1 text-sm font-bold cursor-pointer"\n',
        "                    >\n",
        "                      ✕\n",
        "                    </button>\n",
        "                  </div>\n",
        "                </div>\n",
        "\n",
        "                {/* Status Hero */}\n",
        "                <div className={`p-4 rounded-xl border flex flex-col sm:flex-row justify-between items-center gap-4 ${\n",
        "                  extensionStatus?.connected\n",
        "                    ? 'border-emerald-500/40 bg-emerald-950/20 shadow-[0_0_20px_rgba(16,185,129,0.15)]'\n",
        "                    : 'border-red-500/40 bg-red-950/20 shadow-[0_0_20px_rgba(239,68,68,0.15)]'\n",
        "                }`}>\n",
        "                  <div>\n",
        '                    <div className="flex items-center gap-2">\n',
        '                      <span className="font-bold text-sm text-slate-100">Extension Bridge Connection:</span>\n',
        "                      <span className={`px-2 py-0.5 rounded text-[11px] font-bold border ${\n",
        "                        extensionStatus?.connected\n",
        "                          ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50'\n",
        "                          : 'bg-red-500/20 text-red-400 border-red-500/50'\n",
        "                      }`}>\n",
        "                        {extensionStatus?.connected ? '🟢 CONNECTED' : '🔌 DISCONNECTED'}\n",
        "                      </span>\n",
        "                    </div>\n",
        '                    <div className="text-[11px] text-slate-400 mt-1 font-sans">\n',
        '                      WebSocket Endpoint: <code className="text-argon">ws://127.0.0.1:8000/ws/bridge</code>\n',
        "                    </div>\n",
        "                  </div>\n",
        "                </div>\n",
        "\n",
        "                {/* 3 Step Installation Guide */}\n",
        '                <div className="p-4 rounded-lg bg-surface/80 border border-border flex flex-col gap-3 font-sans">\n',
        '                  <div className="text-[12px] font-bold text-argon uppercase tracking-wider font-mono flex items-center gap-2">\n',
        "                    <span>📋 Quick Setup Instructions</span>\n",
        "                  </div>\n",
        '                  <div className="flex flex-col gap-2.5 text-xs text-slate-300">\n',
        '                    <div className="flex items-start gap-2.5">\n',
        '                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-argon/20 text-argon border border-argon/40 flex items-center justify-center font-bold font-mono text-[11px]">1</span>\n',
        "                      <div>\n",
        '                        Open <code className="bg-obsidian px-1.5 py-0.5 rounded text-argon font-mono text-[11px]">chrome://extensions</code> in Google Chrome and enable <b className="text-slate-100">Developer mode</b> (toggle top-right).\n',
        "                      </div>\n",
        "                    </div>\n",
        '                    <div className="flex items-start gap-2.5">\n',
        '                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-argon/20 text-argon border border-argon/40 flex items-center justify-center font-bold font-mono text-[11px]">2</span>\n',
        "                      <div>\n",
        '                        Click <b className="text-slate-100">Load unpacked</b> and select the <code className="bg-obsidian px-1.5 py-0.5 rounded text-ambercore font-mono text-[11px]">extension/</code> folder located in your AIVCO project directory.\n',
        "                      </div>\n",
        "                    </div>\n",
        '                    <div className="flex items-start gap-2.5">\n',
        '                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-argon/20 text-argon border border-argon/40 flex items-center justify-center font-bold font-mono text-[11px]">3</span>\n',
        "                      <div>\n",
        '                        Keep <b className="text-slate-100">claude.ai</b>, <b className="text-slate-100">chatgpt.com</b>, <b className="text-slate-100">gemini.google.com</b>, or <b className="text-slate-100">grok.com</b> open in any tab.\n',
        "                      </div>\n",
        "                    </div>\n",
        "                  </div>\n",
        "                </div>\n",
        "\n",
        "                {/* Detected Tabs */}\n",
        "                <div>\n",
        '                  <div className="text-[11px] font-bold text-slate-300 uppercase tracking-wider mb-2.5 flex items-center gap-2">\n',
        '                    <span className="w-1.5 h-1.5 rounded-full bg-argon"></span>\n',
        "                    <span>Detected AI Tabs</span>\n",
        "                  </div>\n",
        '                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">\n',
        "                    {[\n",
        "                      { key: 'claude', name: 'Claude', url: 'https://claude.ai' },\n",
        "                      { key: 'chatgpt', name: 'ChatGPT', url: 'https://chatgpt.com' },\n",
        "                      { key: 'gemini', name: 'Gemini', url: 'https://gemini.google.com/app' },\n",
        "                      { key: 'grok', name: 'Grok', url: 'https://grok.com' }\n",
        "                    ].map(prov => {\n",
        "                      const isTabOpen = Boolean(extensionStatus?.tabs?.[prov.key]);\n",
        "                      return (\n",
        '                        <div key={prov.key} className="p-2.5 rounded-lg bg-surface/80 border border-border flex flex-col gap-1.5">\n',
        '                          <div className="flex justify-between items-center">\n',
        '                            <span className="font-bold text-slate-100 text-[11px]">{prov.name}</span>\n',
        "                            <span className={`w-2 h-2 rounded-full ${isTabOpen ? 'bg-emerald-400 shadow-[0_0_6px_#10b981]' : 'bg-slate-600'}`}></span>\n",
        "                          </div>\n",
        "                          <span className={`text-[10px] ${isTabOpen ? 'text-emerald-400 font-bold' : 'text-slate-500'}`}>\n",
        "                            {isTabOpen ? '🟢 Active Tab' : '⚪ Not Detected'}\n",
        "                          </span>\n",
        "                        </div>\n",
        "                      );\n",
        "                    })}\n",
        "                  </div>\n",
        "                </div>\n",
        "\n",
        "                {/* Footer */}\n",
        '                <div className="border-t border-border/80 pt-3 flex justify-between items-center text-[11px] text-slate-400 font-sans">\n',
        "                  <span>AIVCO dispatches prompts directly through the authentic user tab without bot restrictions.</span>\n",
        "                  <button\n",
        '                    type="button"\n',
        "                    onClick={() => setShowVaultModal(false)}\n",
        '                    className="px-4 py-1.5 rounded bg-argon text-obsidian font-bold hover:bg-argon/90 font-mono text-xs cursor-pointer"\n',
        "                  >\n",
        "                    Done\n",
        "                  </button>\n",
        "                </div>\n",
        "              </div>\n",
        "            </div>\n",
        "          )}\n",
        "\n",
    ]
    lines[idx_modal:idx_modal_end] = new_modal_lines
    print(f"Updated modal from index {idx_modal} to {idx_modal_end}")

with open("index.html", "w", encoding="utf-8") as f:
    f.writelines(lines)
print("Successfully wrote updated index.html!")

