#!/usr/bin/env python3
"""Screenshot slides of a deck HTML: shoot.py <deck.html> <outdir> [slide numbers...] (default all)."""
import sys, os, asyncio
from playwright.async_api import async_playwright


async def main():
    deck, out = sys.argv[1], sys.argv[2]
    want = [int(x) for x in sys.argv[3:]]
    os.makedirs(out, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1280, "height": 720})
        await pg.goto("file://" + os.path.abspath(deck))
        await pg.wait_for_function("document.body.dataset.fitted==='1'", timeout=20000)
        await pg.evaluate("document.getElementById('hud').style.display='none'")
        n = await pg.evaluate("S.length")
        for k in (want or range(1, n + 1)):
            await pg.evaluate(f"go({k-1})")
            await pg.wait_for_timeout(150)
            await pg.screenshot(path=f"{out}/s{k:02d}.png")
        print("slides", n)
        await b.close()

asyncio.run(main())
