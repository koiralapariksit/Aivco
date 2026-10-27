"""
[DEPRECATED] scripts/login_manager.py
This script has been deprecated following the architecture pivot to
AIVCO Native Chrome Extension Bridge (Manifest V3).
Playwright-based scraping and directory locks have been removed.
Please load the unpacked extension in chrome://extensions.
"""

import sys

def main():
    print("[DEPRECATED] login_manager.py is no longer used.")
    print("AIVCO now uses the Native Chrome Extension Bridge (Manifest V3).")
    print("Please install the extension from the 'extension/' directory.")

if __name__ == "__main__":
    main()
