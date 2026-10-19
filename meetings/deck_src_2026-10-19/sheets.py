#!/usr/bin/env python3
"""sheets.py — stack slide screenshots two per image into sheets/pNN.png."""
import glob, os
from PIL import Image
D = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/"
os.makedirs(D + "sheets", exist_ok=True)
for f in glob.glob(D + "sheets/*.png"):
    os.remove(f)
shots = sorted(glob.glob(D + "shots/s*.png"))
for k in range(0, len(shots), 2):
    ims = [Image.open(p) for p in shots[k:k + 2]]
    W = max(i.width for i in ims)
    out = Image.new("RGB", (W, sum(i.height for i in ims)), "white")
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height
    out.save(D + f"sheets/p{k // 2 + 1:02d}.png")
print(len(shots), "shots")
