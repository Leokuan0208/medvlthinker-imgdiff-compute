#!/usr/bin/env python3
"""kvasir_x1_neardup_scan.py -- is the kvasir_x1 cell really held out, or just id-held-out?

The cell excludes the 1,052 image IDS the head trained on, and the id-level intersection is 0.
But these are endoscopy VIDEO FRAMES: two frames seconds apart in one procedure get different ids
and are near-identical pixels.  Id-matching cannot see that, so the "3,006 disjoint images" claim
needs a perceptual check before the cell is called held-out.

Uses a 16x16 average hash (downsample to greyscale, threshold at the mean) and reports the Hamming
distance distribution between every kept image and its nearest burned image.
"""
import argparse, json, os, glob
import numpy as np

BURNED = "/data/dan/dataset/kvasir_vqa_x1/images"
CELL = "/data/dan/dataset/kvasir_x1_cell/kvasir_x1_open.json"


def ahash(p, n=16):
    from PIL import Image
    with Image.open(p) as im:
        a = np.asarray(im.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float32)
    return (a > a.mean()).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=12,
                    help="Hamming distance over 256 bits below which two frames are near-duplicates")
    ap.add_argument("--out", default="results/cascade_methods/artifacts/kvasir_x1_neardup_2026-08-19.json")
    A = ap.parse_args()

    burned = sorted(glob.glob(os.path.join(BURNED, "*.jpg")))
    rows = json.load(open(CELL))
    kept = sorted({r["img_path"] for r in rows})
    print(f"burned {len(burned)} | kept {len(kept)}", flush=True)

    B = np.stack([ahash(p) for p in burned])
    print("  hashed burned set", flush=True)
    K, kp = [], []
    for i, p in enumerate(kept):
        try:
            K.append(ahash(p)); kp.append(p)
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            print(f"  hashed {i+1}/{len(kept)} kept", flush=True)
    K = np.stack(K)

    # Hamming via matrix product on {0,1}
    Bf, Kf = B.astype(np.int8), K.astype(np.int8)
    nearest = np.empty(len(Kf), dtype=np.int32)
    for i in range(0, len(Kf), 256):
        blk = Kf[i:i + 256]
        d = (blk[:, None, :] != Bf[None, :, :]).sum(-1)
        nearest[i:i + 256] = d.min(1)
    dup = int((nearest <= A.threshold).sum())
    art = {"title": "kvasir_x1: perceptual near-duplicate scan against the burned training images",
           "date": "2026-08-19", "no_fabricated_numbers": True,
           "method": "16x16 average hash, 256 bits, Hamming distance to the nearest burned image",
           "n_burned": len(burned), "n_kept": len(kp), "threshold_bits": A.threshold,
           "near_duplicates": dup, "near_duplicate_rate": dup / max(len(kp), 1),
           "nearest_distance_percentiles": {p: float(np.percentile(nearest, p))
                                            for p in (0, 1, 5, 25, 50)},
           "verdict": ("NOT a held-out-image cell in the perceptual sense" if dup / max(len(kp), 1) > 0.02
                       else "held-out in both id and perceptual terms at this threshold")}
    # write the exclusion list so the cell can actually be cleaned, not just diagnosed
    bad = sorted({os.path.splitext(os.path.basename(kp[i]))[0]
                  for i in np.where(nearest <= A.threshold)[0]})
    art["excluded_image_ids"] = bad
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"\n  near-duplicates (<= {A.threshold} bits): {dup}/{len(kp)} = {dup/max(len(kp),1):.2%}")
    print(f"  nearest-distance percentiles: {art['nearest_distance_percentiles']}")
    print(f"  {art['verdict']}\n  wrote {A.out}")


if __name__ == "__main__":
    main()
