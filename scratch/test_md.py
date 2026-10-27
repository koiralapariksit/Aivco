import re
import html

def extract_clean_markdown(html_str: str) -> str:
    blocks = {}
    counter = 0

    def code_repl(m):
        nonlocal counter
        lang = m.group(1) or ""
        raw_code = m.group(2)
        code_text = html.unescape(raw_code).strip()
        token = f"___AIVCO_CODE_BLOCK_{counter}___"
        blocks[token] = f"\n```{lang}\n{code_text}\n```\n"
        counter += 1
        return token

    # Replace <pre><code ...>...</code></pre>
    res = re.sub(
        r'<pre[^>]*><code(?: class="[^"]*language-(\w+)[^"]*")?[^>]*>(.*?)</code></pre>',
        code_repl,
        html_str,
        flags=re.DOTALL,
    )
    # Replace <pre>...</pre> without <code>
    def pre_only_repl(m):
        nonlocal counter
        raw_code = m.group(1)
        code_text = html.unescape(raw_code).strip()
        token = f"___AIVCO_CODE_BLOCK_{counter}___"
        blocks[token] = f"\n```\n{code_text}\n```\n"
        counter += 1
        return token

    res = re.sub(r'<pre[^>]*>(?!<code)(.*?)</pre>', pre_only_repl, res, flags=re.DOTALL)
    
    # Strip HTML tags from prose
    res = re.sub(r"<[^>]+>", "", res)
    res = html.unescape(res)

    # Restore code blocks intact
    for token, fence in blocks.items():
        res = res.replace(token, fence)

    res = re.sub(r"\n{3,}", "\n\n", res)
    return res.strip()

sample = '<p>Here is your code:</p><pre><code class="language-html">&lt;button class="btn"&gt;Click Me&lt;/button&gt;</code></pre><p>Enjoy!</p>'
out = extract_clean_markdown(sample)
print(out)
assert '```html' in out
assert '<button class="btn">Click Me</button>' in out
assert '```' in out
print("\nEXTRACTOR PASSED!")

