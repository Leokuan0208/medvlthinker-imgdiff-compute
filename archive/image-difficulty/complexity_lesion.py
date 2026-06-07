"""REFINE: region size/contrast from SLAKE masks -> comp_lesion_* in complexity.csv.
SLAKE ships an organ segmentation mask per image folder (mask.png). Region size +
contrast is a better difficulty proxy than whole-image texture. Run AFTER complexity.py,
then re-run analyze.py. Expected if the wedge is real: difficulty rises as region
area/contrast FALL (small/faint region = harder) -> NEGATIVE rho, STRONGER than texture.
"""
import csv, os
import numpy as np
from PIL import Image

DIFF_CSV, COMP_CSV = "difficulty.csv", "complexity.csv"

def region_features(image_path):
    folder = os.path.dirname(image_path)
    mpath = os.path.join(folder, "mask.png")
    if not os.path.exists(mpath):
        return None
    m = np.asarray(Image.open(mpath))
    if m.ndim == 3:
        m = m[..., 0]
    fg = m > 0
    area = float(fg.mean())
    if area in (0.0, 1.0):
        contrast = 0.0
    else:
        g = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
        contrast = float(abs(g[fg].mean() - g[~fg].mean()))
    return area, contrast

def main():
    print("start", flush=True)
    comp = {r["image_id"]: r for r in csv.DictReader(open(COMP_CSV))}
    paths = {}
    for r in csv.DictReader(open(DIFF_CSV)):
        paths.setdefault(r["image_id"], r["image_path"])
    print(f"{len(comp)} complexity rows, {len(paths)} images to probe", flush=True)
    feats, probe = {}, 0
    for iid, p in paths.items():
        f = region_features(p)
        if probe < 5:
            print(f"[probe] {iid}: path={p} mask={'yes' if f else 'NO'}" +
                  (f" area={f[0]:.4f} contrast={f[1]:.1f}" if f else ""), flush=True)
            probe += 1
        if f:
            feats[iid] = f
    if not feats:
        print(">>> no masks resolved — paste a [probe] path line and I'll fix it", flush=True)
        return
    med_a = float(np.median([v[0] for v in feats.values()]))
    med_c = float(np.median([v[1] for v in feats.values()]))
    for iid in comp:
        a, c = feats.get(iid, (med_a, med_c))
        comp[iid]["comp_lesion_area"] = round(a, 6)
        comp[iid]["comp_lesion_contrast"] = round(c, 4)
    fields = list(next(iter(comp.values())).keys())
    with open(COMP_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(comp.values())
    print(f"updated {COMP_CSV}: mask coverage {len(feats)}/{len(paths)} images", flush=True)
    print("now run: python3 analyze.py", flush=True)

if __name__ == "__main__":
    main()
