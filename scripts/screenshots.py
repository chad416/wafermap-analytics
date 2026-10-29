"""Capture README screenshots from a running stack.

    python scripts/screenshots.py --base http://localhost:8080 [--channel msedge]

Uses Playwright with an already-installed browser channel (Edge / Chrome), so no browser
download is needed. Pages are addressed by URL (filters live in the query string), which keeps
the screenshots reproducible. The lot and wafer shown are picked through the API rather than
hard-coded, so the script works on any ingested database.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base}/api/v1{path}", timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--channel", default="msedge")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    OUT.mkdir(parents=True, exist_ok=True)

    # A lot that has several predicted defect types (more interesting than an all-clean lot).
    lots = get(base, "/lots?pattern=Edge-Ring&source=true&page_size=50")["items"]
    lot = max(
        lots, key=lambda x: (len([k for k in x["pattern_counts"] if k != "none"]), x["wafer_count"])
    )
    scratch = get(base, "/wafers?pattern=Scratch&source=true&page_size=1")["items"][0]

    shots = [
        ("01_overview.png", "/", 1400, 1110),
        ("02_lots_filtered.png", "/lots?pattern=Scratch", 1400, 900),
        ("03_lot_drilldown.png", f"/lots/{lot['name']}", 1400, 1310),
        ("05_api_docs.png", "/docs", 1400, 1000),
    ]
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel)
        for name, path, w, h in shots:
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
            page.goto(base + path, wait_until="networkidle")
            page.wait_for_timeout(800)  # let charts / canvases settle
            page.screenshot(path=str(OUT / name), full_page=False)
            page.close()
            print("saved", name)

        # Wafer detail with the on-demand baseline result next to the stored CNN prediction.
        page = browser.new_page(viewport={"width": 1400, "height": 820})
        page.goto(f"{base}/wafers/{scratch['id']}", wait_until="networkidle")
        page.click("text=Classify with baseline")
        page.wait_for_selector("text=ms server-side")
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "04_wafer_detail.png"))
        page.close()
        print("saved 04_wafer_detail.png")

        # Upload flow: classify a synthetic sample through the real UI.
        page = browser.new_page(viewport={"width": 1400, "height": 950})
        page.goto(base + "/classify", wait_until="networkidle")
        sample = (
            Path(__file__).resolve().parents[1] / "frontend" / "public" / "samples" / "donut.csv"
        )
        page.set_input_files("input[type=file]", str(sample))
        page.wait_for_selector("text=Prediction:")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "06_upload_classify.png"))
        print("saved 06_upload_classify.png")
        browser.close()


if __name__ == "__main__":
    main()
