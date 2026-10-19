"""crop_results.py — crop one RESULTS figure per paper (drawings/images above its caption) to paper_figs/<key>_res_fig<N>.png.
Usage: python -I crop_results.py [manual.json]  (manual clip rects in PDF points override the automatic box)."""
import io, json, os, re, sys
import fitz
from PIL import Image

P = "/data/dan/literature/deck_papers_2026-10-19/"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
FIGS = [  # key, pdf, page (1-based) to search first, figure number
    ("cobbe", "1_core_Cobbe2021_TrainingVerifiers.pdf", 8, 5),
    ("brown", "1_core_Brown2024_LargeLanguageMonkeys.pdf", 10, 7),
    ("snell", "1_core_Snell2024_ScalingTestTimeCompute.pdf", 10, 4),
    ("wang", "1_core_Wang2023_SelfConsistency.pdf", 6, 2),
    ("usc", "1_core_Chen2023_UniversalSelfConsistency_USC.pdf", 7, 3),
    ("swift", "1_core_Guo2025_SWIFT.pdf", 3, 2),
    ("hsrm", "1_core_Li2026_HSRM.pdf", 7, 4),
    ("mirage", "1_core_Jin2026_VerificationMirage.pdf", 7, 2),
    ("medprob", "1_core_Nourbakhsh2026_MedProb.pdf", 7, 5),
    ("boe", "1_core_Zhang2026_BestOfEvidence.pdf", 8, 3),
]
MANUAL = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
info = json.load(open(OUT + "res_index.json")) if os.path.exists(OUT + "res_index.json") else {}
for key, pdf, pg0, n in FIGS:
    if only and key not in only:
        continue
    doc = fitz.open(P + pdf)
    found = None
    for pg in [pg0] + [p for p in range(pg0 - 2, pg0 + 3) if p != pg0 and 1 <= p <= len(doc)]:
        page = doc[pg - 1]
        for b in page.get_text("blocks"):
            t = " ".join(b[4].split())
            if re.match(rf"^(Figure|Fig\.?)\s*{n}\b", t):
                found = (pg, page, fitz.Rect(b[:4])); break
        if found:
            break
    if not found:
        print(key, "caption not found"); continue
    pg, page, cap = found
    W = page.rect.width
    if key in MANUAL:
        clip = fitz.Rect(MANUAL[key])
    else:
        rects = [fitz.Rect(d["rect"]) for d in page.get_drawings()] + [fitz.Rect(i["bbox"]) for i in page.get_image_info()]
        rects = [r for r in rects if r.y1 <= cap.y0 + 2 and r.y0 >= cap.y0 - 520 and r.width < W * 0.98 and r.height > 1]
        clip = fitz.Rect(rects[0]) if rects else fitz.Rect(36, cap.y0 - 300, W - 36, cap.y0)
        for r in rects[1:]:
            clip |= r
        clip = fitz.Rect(clip.x0 - 4, clip.y0 - 4, clip.x1 + 4, cap.y0 - 1) & page.rect
    pix = page.get_pixmap(dpi=200, clip=clip)
    im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    path = f"{OUT}{key}_res_fig{n}.png"
    im.save(path, optimize=True)
    if os.path.getsize(path) > 250_000:
        im.quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    if os.path.getsize(path) > 250_000:
        im2 = im.resize((int(im.width * 0.75), int(im.height * 0.75)), Image.LANCZOS)
        im2.quantize(colors=128).save(path, optimize=True)
    info[key] = {"pdf": pdf, "page": pg, "figure": n, "caption": " ".join(page.get_text("text", clip=cap).split())[:200],
                 "clip": [round(v, 1) for v in clip], "px": list(Image.open(path).size), "kb": os.path.getsize(path) // 1024}
    print(key, info[key])
json.dump(info, open(OUT + "res_index.json", "w"), indent=1)
