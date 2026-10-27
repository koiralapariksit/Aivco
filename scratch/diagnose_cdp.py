"""
Diagnostic script for Live CDP tabs on port 9222.
Extracts contexts, pages, URLs, titles, and visible input/button elements.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
from playwright.sync_api import sync_playwright

def inspect_element(el):
    try:
        tag = el.evaluate("e => e.tagName.toLowerCase()")
        e_id = el.get_attribute("id") or ""
        e_cls = el.get_attribute("class") or ""
        e_aria = el.get_attribute("aria-label") or ""
        e_testid = el.get_attribute("data-testid") or ""
        e_ph = el.get_attribute("placeholder") or ""
        text = (el.text_content() or "").strip()[:60]
        vis = el.is_visible()
        return {
            "tag": tag,
            "id": e_id,
            "class": " ".join(e_cls.split())[:80],
            "aria_label": e_aria,
            "data_testid": e_testid,
            "placeholder": e_ph,
            "text": text,
            "visible": vis,
        }
    except Exception as exc:
        return {"error": str(exc)}

def dump_dom(page, provider_name):
    print(f"\n=======================================================")
    print(f" DOM DUMP FOR {provider_name.upper()} ({page.url})")
    print(f" Page Title: {page.title()}")
    print(f"=======================================================")
    
    selectors = [
        "textarea",
        "[contenteditable='true']",
        "[contenteditable]",
        "[role='textbox']",
        "input[type='text']",
        "input[type='email']",
        "input:not([type='hidden'])",
        "button",
    ]
    
    seen = set()
    found = []
    for sel in selectors:
        try:
            elements = page.query_selector_all(sel)
            for el in elements:
                try:
                    if not el.is_visible():
                        continue
                    info = inspect_element(el)
                    key = (info.get("tag"), info.get("id"), info.get("aria_label"), info.get("placeholder"), info.get("text"))
                    if key not in seen:
                        seen.add(key)
                        found.append(info)
                except Exception:
                    pass
        except Exception:
            pass
            
    print(f"Visible actionable elements found ({len(found)}):")
    for idx, item in enumerate(found, 1):
        clean_text = repr(item['text'])[1:-1]
        print(f" [{idx}] <{item['tag']}> id='{item['id']}' testid='{item['data_testid']}' aria='{item['aria_label']}' ph='{item['placeholder']}' text='{clean_text}' class='{item['class']}'")

def main():
    with sync_playwright() as p:
        print("[CHECK 1: CONTEXT & PAGE ENUMERATION]")
        browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
        contexts = browser.contexts
        print(f"Total browser.contexts: {len(contexts)}")
        
        all_pages = []
        for c_idx, ctx in enumerate(contexts):
            print(f"\nContext #{c_idx} (pages: {len(ctx.pages)}):")
            for p_idx, page in enumerate(ctx.pages):
                url = page.url
                title = page.title()
                print(f"  Page [{c_idx}:{p_idx}] URL: {url} | Title: '{title}'")
                all_pages.append((c_idx, p_idx, page))

        # Check Claude
        claude_pages = [pg for _, _, pg in all_pages if "claude.ai" in pg.url.lower()]
        if claude_pages:
            dump_dom(claude_pages[0], "Claude")
        else:
            print("\n[!] No open tab matching claude.ai found!")

        # Check ChatGPT
        chatgpt_pages = [pg for _, _, pg in all_pages if "chatgpt.com" in pg.url.lower()]
        if chatgpt_pages:
            dump_dom(chatgpt_pages[0], "ChatGPT")
        else:
            print("\n[!] No open tab matching chatgpt.com found!")

        # Check Gemini
        gemini_pages = [pg for _, _, pg in all_pages if "gemini.google.com" in pg.url.lower()]
        if gemini_pages:
            dump_dom(gemini_pages[0], "Gemini")
        else:
            print("\n[!] No open tab matching gemini.google.com found!")

        # Check Grok
        grok_pages = [pg for _, _, pg in all_pages if "grok.com" in pg.url.lower()]
        if grok_pages:
            dump_dom(grok_pages[0], "Grok")
        else:
            print("\n[!] No open tab matching grok.com found!")

        browser.close()

if __name__ == "__main__":
    main()

