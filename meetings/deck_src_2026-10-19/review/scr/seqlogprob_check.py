"""Is a per-sample sequence log-probability stored in the shipped cap320 T=0.7 pools? Also greedy answer + labels for 12231."""
import json
CK = "/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b"


def find(p, idx=12231):
    for l in open(p):
        d = json.loads(l)
        if int(d["idx"]) == idx:
            return d


g = find(f"{CK}/ckpt_slake_open_lingshu7b.jsonl")
print("greedy", {k: g[k] for k in g if k != "raw_output"})
for t in ("lingshu7bT02_sc8", "lingshu7bT04_sc8", "lingshu7b_sc8", "lingshu7bT10_sc8"):
    d = find(f"{CK}/ckpt_slake_open_{t}.jsonl")
    print(t, "oks", d.get("oks"), "seqlogprob", d.get("seqlogprob"))
for c in ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]:
    n = k = 0
    for l in open(f"{CK}/ckpt_{c}_lingshu7b_sc8.jsonl"):
        d = json.loads(l); n += 1
        s = d.get("seqlogprob")
        if isinstance(s, list) and s and all(v is not None for v in s):
            k += 1
    print(c, k, "/", n)
