#!/usr/bin/env python3
"""overflow.py <deck.html> — report slides whose body content spills out of the body box, or whose body has much empty space."""
import sys, os, asyncio, json
from playwright.async_api import async_playwright

JS = """(k) => {
  go(k);
  const s = S[k]; const b = s.querySelector('.body'); if (!b) return null;
  const br = b.getBoundingClientRect(); let maxB = br.top, maxR = br.left, bad = [];
  for (const e of b.querySelectorAll('*')) { if (e.closest('.katex-mathml')) continue;
    const r = e.getBoundingClientRect(); if (!r.width || !r.height) continue;
    if (r.bottom > maxB) maxB = r.bottom; if (r.right > maxR) maxR = r.right;
    if (r.bottom > br.bottom + 2 || r.right > br.right + 2) bad.push(e.tagName + '.' + (e.className.baseVal ?? e.className) + ' ' + Math.round(r.bottom - br.bottom) + '/' + Math.round(r.right - br.right));
  }
  // sibling blocks (cards, tables, figures) drawn on top of each other
  const blocks = [...b.querySelectorAll('.term,.paper,.box,.kpi,table,svg,ul,.eq,.katex-display')]
    .filter(e => !blocks0(e));
  function blocks0(e){ const p = e.parentElement && e.parentElement.closest('.term,.paper,.box,.kpi,table,svg,ul,.eq,.katex-display'); return p && b.contains(p); }
  for (let i = 0; i < blocks.length; i++) for (let j = i + 1; j < blocks.length; j++) {
    const a = blocks[i].getBoundingClientRect(), c = blocks[j].getBoundingClientRect();
    const ox = Math.min(a.right, c.right) - Math.max(a.left, c.left), oy = Math.min(a.bottom, c.bottom) - Math.max(a.top, c.top);
    if (ox > 4 && oy > 4) bad.push('overlap ' + blocks[i].tagName + '.' + (blocks[i].className.baseVal ?? blocks[i].className) + ' × ' + blocks[j].tagName + '.' + (blocks[j].className.baseVal ?? blocks[j].className) + ' ' + Math.round(ox) + 'x' + Math.round(oy));
  }
  // svg text clipped by its own svg viewport
  for (const sv of b.querySelectorAll('svg')) {
    const sr = sv.getBoundingClientRect();
    for (const tx of sv.querySelectorAll('text')) {
      const r = tx.getBoundingClientRect();
      if (r.right > sr.right + 1 || r.left < sr.left - 1 || r.bottom > sr.bottom + 1) bad.push('svgtext "' + tx.textContent.slice(0, 40) + '"');
    }
  }
  return {n: k + 1, title: s.dataset.title, fill: (maxB - br.top) / br.height, bad: bad.slice(0, 8)};
}"""


async def main():
    deck = sys.argv[1]
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1280, "height": 720})
        await pg.goto("file://" + os.path.abspath(deck))
        await pg.wait_for_function("document.body.dataset.fitted==='1'", timeout=20000)
        await pg.evaluate("document.getElementById('deck').style.transform='none'")
        n = await pg.evaluate("S.length")
        for k in range(n):
            r = await pg.evaluate(JS, k)
            if not r:
                continue
            flag = ("OVER " if r["bad"] else "") + ("SPARSE " if r["fill"] < 0.6 else "")
            if flag:
                print(f"{r['n']:>3} {flag:12s} fill={r['fill']:.2f} {r['title'][:60]}  {r['bad']}")
        await b.close()

asyncio.run(main())
