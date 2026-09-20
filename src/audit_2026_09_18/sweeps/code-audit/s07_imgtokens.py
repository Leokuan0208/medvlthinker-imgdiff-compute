"""s07: decisive check that the probe's forward pass used a DIFFERENT image resolution than
generation. run_openvqa.py default --cap cap320 -> max_pixels 1003520//4 = 250880 -> at most
250880/(28*28)/(2*2) = 80 image tokens. extract_generator_hidden.py hard-codes HIGH_PX = 1003520
-> at most 320 image tokens. n_img_tok is saved in the npz."""
import json, os
import numpy as np

FE = "/home/jamesyang/medvlthinker-imgdiff-compute/feats_hidden"
cells = ["generator_eval_finelayer", "generator_eval_finelayer_pathvqa_open",
         "generator_eval_finelayer_radimagenet_open", "generator_eval_finelayer_omnimed_open",
         "generator_eval_finelayer_gemex_open", "generator_train_finelayer"]
print("cap320 ceiling = 80 image tokens ; fullres(HIGH_PX) ceiling = 320 image tokens")
print("stem | meta.max_pixels | n_img_tok min/mean/median/max | frac>80")
for st in cells:
    p = f"{FE}/{st}.npz"
    mp = f"{FE}/{st}.meta.json"
    if not os.path.exists(p):
        print(f"{st}: MISSING"); continue
    m = json.load(open(mp))
    z = np.load(p)
    if "n_img_tok" not in z.files:
        print(f"{st}: no n_img_tok key ({z.files})"); continue
    t = z["n_img_tok"].astype(int)
    print(f"{st:45s} max_pixels={m.get('max_pixels')} sys={str(m.get('sys_prompt'))[:38]!r}")
    print(f"     n_img_tok  min {t.min()}  mean {t.mean():.1f}  median {int(np.median(t))}  "
          f"max {t.max()}  frac>80 {float((t > 80).mean()):.4f}")
    del z
