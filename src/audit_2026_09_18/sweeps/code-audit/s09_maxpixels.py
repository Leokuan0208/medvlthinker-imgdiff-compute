import json, os
FE = "/home/jamesyang/medvlthinker-imgdiff-compute/feats_hidden"
stems = ["generator_train_finelayer", "generator_eval_finelayer",
         "generator_eval_finelayer_pathvqa_open", "generator_eval_finelayer_radimagenet_open",
         "generator_eval_finelayer_kvasir_x1_open", "generator_eval_finelayer_omnimed_open",
         "generator_eval_finelayer_vqamed_open", "generator_eval_finelayer_gemex_open",
         "generator_train_s0of2", "generator_train_s1of2"]
print("generation cap320 max_pixels = 250880")
for s in stems:
    p = f"{FE}/{s}.meta.json"
    if not os.path.exists(p):
        print(f"{s:48s} MISSING"); continue
    m = json.load(open(p))
    print(f"{s:48s} max_pixels={m.get('max_pixels')} mode={m.get('mode')} "
          f"n={m.get('n')} n_failed={m.get('n_failed')} model={str(m.get('model'))[-28:]}")
