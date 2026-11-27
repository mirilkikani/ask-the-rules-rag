"""Keep the Streamlit Community Cloud demo awake.

Free Community Cloud apps go to sleep after about 12 hours without visitors. A plain HTTP request
does not count (it only gets a static page), so this script opens the app in a real headless
browser. If the app is asleep, it clicks "Yes, get this app back up!" and waits until it runs.
Run by .github/workflows/keep_awake.yml every 6 hours. No Gemini call is made.

Usage:
    python scripts/keep_awake.py [url]
"""
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://ask-the-rules.streamlit.app"
WAKE_BUTTON = "Yes, get this app back up!"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(8_000)

        button = page.get_by_role("button", name=WAKE_BUTTON)
        if button.count() > 0:
            print(f"Asleep -> clicking '{WAKE_BUTTON}'")
            button.first.click()
            page.wait_for_timeout(90_000)        # waking up and loading the model takes about a minute
            status = "woken up"
        else:
            page.wait_for_timeout(20_000)        # stay a moment, so the visit counts as traffic
            status = "already awake"

        still_asleep = page.get_by_role("button", name=WAKE_BUTTON).count() > 0
        browser.close()
    print(f"{URL}: {status}" + (" (but the wake-up button is still there)" if still_asleep else ""))
    sys.exit(1 if still_asleep else 0)


if __name__ == "__main__":
    main()
