import re

with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

scripts = re.findall(r'<script type=["\']text/babel["\']>(.*?)</script>', html, re.DOTALL)
print(f"Found {len(scripts)} babel scripts")
if scripts:
    with open("scratch/babel_content.jsx", "w", encoding="utf-8") as out:
        out.write(scripts[0])
    print(f"Babel script length: {len(scripts[0])} chars")

