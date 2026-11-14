#!/usr/bin/env python3
"""Record a short browser demo of Character Forge as an MP4."""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = os.environ.get("FORGE_URL", "http://127.0.0.1:7860/")
PHRASE = "digital loneliness"
VIDEO_DIR = ROOT / "outputs" / "demo-raw"
OUTPUT = ROOT / "outputs" / "character-forge-demo.mp4"

CURSOR_JS = """
(() => {
  const cursor = document.createElement('div');
  cursor.style.cssText = [
    'position:fixed','width:18px','height:18px','border-radius:50%',
    'background:rgba(178,58,46,.25)','border:2px solid rgba(178,58,46,.9)',
    'z-index:2147483647','pointer-events:none','transform:translate(-50%,-50%)',
    'transition:width .08s,height .08s,background .08s','left:-40px','top:-40px'
  ].join(';');
  document.body.appendChild(cursor);
  document.addEventListener('mousemove', (e) => {
    cursor.style.left = e.clientX + 'px';
    cursor.style.top = e.clientY + 'px';
  });
  const down = () => { cursor.style.width='30px'; cursor.style.height='30px';
    cursor.style.background='rgba(178,58,46,.12)'; };
  const up = () => { cursor.style.width='18px'; cursor.style.height='18px';
    cursor.style.background='rgba(178,58,46,.25)'; };
  document.addEventListener('mousedown', down);
  document.addEventListener('mouseup', up);
})();
"""


def wait_for_app(timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(URL, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"App is not reachable at {URL}; run `python app.py` first.")


def move_click(page, locator) -> None:
    box = locator.bounding_box()
    if not box:
        locator.click()
        return
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y, steps=18)
    page.wait_for_timeout(120)
    page.mouse.click(x, y)


def main() -> int:
    wait_for_app()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for stale in VIDEO_DIR.glob("*.webm"):
        stale.unlink()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": 1440, "height": 1000},
        )
        page = context.new_page()
        page.goto(URL, wait_until="load", timeout=60000)
        page.wait_for_timeout(2500)
        page.evaluate(CURSOR_JS)
        page.mouse.move(720, 500)
        page.wait_for_timeout(600)

        phrase_box = page.get_by_label("English phrase")
        move_click(page, phrase_box)
        page.keyboard.type(PHRASE, delay=28)
        page.wait_for_timeout(500)

        move_click(page, page.get_by_role("button", name="Forge character"))
        page.wait_for_selector("img[alt*='LLM']", timeout=25000)
        page.wait_for_timeout(2500)

        move_click(page, page.get_by_role("button", name="Remix"))
        page.wait_for_timeout(5200)

        thumbs = page.locator("img[alt*='LLM']")
        if thumbs.count() >= 3:
            move_click(page, thumbs.nth(2))
            page.wait_for_timeout(2200)

        main_image = page.locator("img").first
        box = main_image.bounding_box()
        if box:
            page.mouse.move(
                box["x"] + box["width"] - 60,
                box["y"] + box["height"] - 80,
                steps=20,
            )
        page.wait_for_timeout(2500)

        video = page.video
        context.close()
        browser.close()
        webm = Path(video.path())

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            exe,
            "-y",
            "-i",
            str(webm),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            str(OUTPUT),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Demo saved to {OUTPUT} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
