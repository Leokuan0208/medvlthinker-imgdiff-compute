import json
F = "/home/jamesyang/medvlthinker-imgdiff-compute/feats_hidden/"
for st in ["generator_eval_qwen_pathvqa_open", "generator_train_qwen_slake_open_train",
           "generator_eval_medgemma_pathvqa_open", "generator_eval_finelayer_pathvqa_open"]:
    m = json.load(open(F + st + ".meta.json"))
    print(st, {k: (v if not isinstance(v, list) else "list[%d]" % len(v)) for k, v in m.items()})
    print("  ", m["rows"][0])
    print("  ", m["rows"][-1])
