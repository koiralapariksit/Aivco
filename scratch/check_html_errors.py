import asyncio
from playwright.async_api import async_playwright
from pathlib import Path

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        errors = []
        page.on("pageerror", lambda err: errors.append(f"PAGE_ERROR: {err}"))
        page.on("console", lambda msg: print(f"CONSOLE [{msg.type}]: {msg.text}") if msg.type in ["error", "warning"] else None)

        file_url = Path("index.html").resolve().as_uri()
        print(f"Navigating to {file_url}...")
        await page.goto(file_url, wait_until="networkidle", timeout=15000)
        await page.wait_for_timeout(2000)

        root_html = await page.inner_html("#root")
        print(f"Root innerHTML length: {len(root_html)}")
        if errors:
            print("Errors detected:")
            for e in errors:
                print(e)
        else:
            print("No runtime page errors detected.")
        await browser.close()

asyncio.run(main())

