import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import browser_agent

# Test 1: Markdown code extraction
html_sample = '<p>Check this code:</p><pre><code class="language-python">def hello():\n    return 42</code></pre>'
md_res = browser_agent._extract_clean_markdown(html_sample)
print('Clean Markdown:\n', md_res)
assert '```python' in md_res
assert 'def hello():' in md_res

# Test 2: UniversalWebAgent status
st = browser_agent.universal_web_agent.status()
print('Agent status:', st)
assert 'profile_dir' in st

print('\nBROWSER AGENT BASIC CHECKS PASSED!')

