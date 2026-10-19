"""crop_figs.py — crop each paper's overview figure (drawings/images above its caption) to paper_figs/<key>_fig<N>.png.
Manual clip rects (PDF points, x0,y0,x1,y1) override the automatic box where given."""
import io, json, os, re, sys
import fitz
from PIL import Image

P = "/data/dan/literature/domain_package_2026-09-16/papers/"
X = "/home/jamesyang/.claude/jobs/61c9eff9/tmp/arxiv_pdfs/"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/paper_figs/"
os.makedirs(OUT, exist_ok=True)
FIGS = [  # key, pdf, page (1-based), figure number
    ("cobbe", P + "A_cobbe2021verifiers.pdf", 7, 4),
    ("brown", P + "A_brown2024monkeys.pdf", 2, 1),
    ("snell", P + "A_snell2024scaling.pdf", 8, 2),
    ("wang", P + "E_wang2023selfconsistency.pdf", 2, 1),
    ("usc", P + "E_chen2023usc.pdf", 2, 1),
    ("qprobe", P + "C_li2024qprobe.pdf", 2, 1),
    ("swift", X + "2505.12225.pdf", 2, 1),
    ("hsrm", P + "C_li2026hsrm.pdf", 3, 1),
    ("case", X + "2608.17124.pdf", 8, 1),
    ("mirage", P + "B_jin2026verificationmirage.pdf", 2, 1),
    ("medprob", P + "C_nourbakhsh2026medprob.pdf", 5, 2),
    ("dualread", X + "2609.06419.pdf", 3, 2),
    ("wed", P + "E_hagen2026wed.pdf", 3, 1),
    ("boe", P + "B_zhang2026bestofevidence.pdf", 1, 1),
]
MANUAL = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
info = {}
for key, pdf, pg, n in FIGS:
    page = fitz.open(pdf)[pg - 1]
    cap = None
    for b in page.get_text("blocks"):
        t = " ".join(b[4].split())
        if re.match(rf"^(Figure|Fig\.?)\s*{n}\b", t):
            cap = fitz.Rect(b[:4]); break
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
    path = f"{OUT}{key}_fig{n}.png"
    im.save(path, optimize=True)
    if os.path.getsize(path) > 250_000:
        im.quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    if os.path.getsize(path) > 250_000:
        im2 = im.resize((int(im.width * 0.75), int(im.height * 0.75)), Image.LANCZOS)
        im2.quantize(colors=128).save(path, optimize=True)
    info[key] = {"pdf": os.path.basename(pdf), "page": pg, "figure": n, "clip": [round(v, 1) for v in clip],
                 "px": list(Image.open(path).size), "kb": os.path.getsize(path) // 1024}
    print(key, info[key])
json.dump(info, open(OUT + "figs_index.json", "w"), indent=1)
