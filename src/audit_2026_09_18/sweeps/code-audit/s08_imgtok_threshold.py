"""s08: Qwen2.5-VL patch_size=14, merge_size=2 -> factor 28; image tokens = (H*W)/784 after
smart_resize caps H*W at max_pixels. So:
    extraction  max_pixels = HIGH_PX      = 1280*784 = 1,003,520  -> ceiling 1280 image tokens
    generation  max_pixels = HIGH_PX // 4 =  320*784 =   250,880  -> ceiling  320 image tokens
Rows whose recorded n_img_tok > 320 CANNOT have been produced at cap320."""
import json, os
import numpy as np

FE = "/home/jamesyang/medvlthinker-imgdiff-compute/feats_hidden"
st = "generator_eval_finelayer_pathvqa_open"
m = json.load(open(f"{FE}/{st}.meta.json"))
z = np.load(f"{FE}/{st}.npz")
t = z["n_img_tok"].astype(int)
print("meta.max_pixels        :", m.get("max_pixels"))
print("run_openvqa cap320 px  : 250880   (HIGH_PX//CAP_DIV['cap320'], run_openvqa.py:51-52,79)")
print("ratio                  :", m.get("max_pixels") / 250880)
print()
print("n_img_tok  n=%d  min %d  mean %.1f  median %d  p90 %d  max %d"
      % (len(t), t.min(), t.mean(), int(np.median(t)), int(np.percentile(t, 90)), t.max()))
for thr in (320, 640, 1280):
    print(f"   frac(n_img_tok > {thr:4d}) = {float((t > thr).mean()):.4f}")
print()
print("mean image tokens at extraction :", round(float(t.mean()), 1))
print("implied mean at cap320 (cap at 320, else same):",
      round(float(np.minimum(t, 320).mean()), 1))
print("=> extra image tokens per candidate forward pass:",
      round(float(t.mean() - np.minimum(t, 320).mean()), 1))
