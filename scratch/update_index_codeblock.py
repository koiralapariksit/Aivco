with open("index.html", "r", encoding="utf-8") as f:
    content = f.read()

target1 = """      useEffect(() => {
        if (codeRef.current && window.hljs) {
          try {
            window.hljs.highlightElement(codeRef.current);
          } catch (e) {
            // fallback
          }
        }
      }, [code, language]);"""

repl1 = """      useEffect(() => {
        if (codeRef.current) {
          codeRef.current.textContent = code;
          if (window.hljs) {
            try {
              codeRef.current.removeAttribute('data-highlighted');
              window.hljs.highlightElement(codeRef.current);
            } catch (e) {
              // fallback
            }
          }
        }
      }, [code, language]);"""

target2 = """          // Auto-open artifact if response contains HTML document or full code
          if (data.response && (data.response.includes('<!DOCTYPE html') || data.response.includes('<html'))) {
            const htmlMatch = data.response.match(/```(?:html)?\\n([\\s\\S]*?)```/i);
            if (htmlMatch && htmlMatch[1]) {
              setCurrentArtifact({
                title: 'Live Web Artifact',
                lang: 'html',
                code: htmlMatch[1]
              });
              setArtifactTab('preview');
            }
          }"""

repl2 = """          // Auto-open artifact if response contains HTML document or ```html code block
          if (data.response) {
            const htmlBlockRegex = /```(?:html|htm|xml)?\\n([\\s\\S]*?)```/gi;
            let m;
            let bestCode = null;
            let maxLen = 0;
            while ((m = htmlBlockRegex.exec(data.response)) !== null) {
              const candidate = m[1].trim();
              if (candidate.includes('<html') || candidate.includes('<!DOCTYPE') || candidate.includes('<body') || candidate.includes('<style')) {
                if (candidate.length > maxLen) {
                  maxLen = candidate.length;
                  bestCode = candidate;
                }
              }
            }

            if (!bestCode && (data.response.includes('<!DOCTYPE html') || data.response.includes('<html'))) {
              const anyBlockMatch = data.response.match(/```([a-zA-Z0-9_\\-+]*)\\n([\\s\\S]*?)```/);
              if (anyBlockMatch && anyBlockMatch[2] && (anyBlockMatch[2].includes('<html') || anyBlockMatch[2].includes('<!DOCTYPE'))) {
                bestCode = anyBlockMatch[2].trim();
              }
            }

            if (bestCode && bestCode.length > 30) {
              setCurrentArtifact({
                title: 'Live Web Artifact',
                lang: 'html',
                code: bestCode
              });
              setArtifactTab('preview');
              setIframeKey(k => k + 1);
            }
          }"""

# Normalize \r\n to \n
content_norm = content.replace("\r\n", "\n")
target1_norm = target1.replace("\r\n", "\n")
target2_norm = target2.replace("\r\n", "\n")

assert target1_norm in content_norm, "target1 not found"
assert target2_norm in content_norm, "target2 not found"

content_norm = content_norm.replace(target1_norm, repl1)
content_norm = content_norm.replace(target2_norm, repl2)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(content_norm)

print("index.html successfully updated!")

