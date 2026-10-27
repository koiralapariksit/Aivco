import urllib.request
import urllib.parse
import re
import html

q = "which football player has won most ballon dor"
encoded_q = urllib.parse.quote(q)

# Test DuckDuckGo HTML
try:
    url = f"https://html.duckduckgo.com/html/?q={encoded_q}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    res = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", errors="ignore")
    # Result snippets in html.duckduckgo.com
    # Snippet usually in class="result__snippet"
    snippets = re.findall(r'class="result__snippet[^"]*"[^>]*>(.*?)</a>', res, re.DOTALL)
    if not snippets:
        # try lite or general pattern
        snippets = re.findall(r'class="result-snippet"[^>]*>(.*?)</td>', res, re.DOTALL)
    print(f"DDG HTML Snippets found: {len(snippets)}")
    for s in snippets[:3]:
        clean = html.unescape(re.sub(r'<[^>]+>', '', s)).strip()
        print(" -", clean[:150])
except Exception as e:
    print("DDG HTML error:", e)

