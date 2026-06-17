#!/usr/bin/env python3
"""
router_cost_prefill.py - prefill-INCLUSIVE compute for the frozen cascade (review-proof).

The decode-only proxy was OPTIMISTIC: it ignored prefill, but a cascade pays the 7B's
prefill (text + expanded vision tokens) on EVERY question and the 32B's prefill again on
escalated ones. This measures the true prompt length per question with the Qwen processor
(same build() as run_7b_vllm.py, same HIGH_PX cap) and counts FLOPs = 2*N*(prefill+decode)
for each stage, with an optional ViT term. Decode lengths come from gen_tokens in the labels.

FLOPs model (stated, adjustable):
  one model run on a question  =  2 * N_backbone * (P + G)        [+ 2 * N_vit * patches]
  P = prompt tokens (incl vision),  G = generated tokens,  patches = pre-merge ViT patches.
  cascade(q) = run_7B(q)  +  [escalated] run_32B(q)      always-32B(q) = run_32B(q)

Reports per dataset and aggregate: 32B-call rate (mechanism-agnostic headline), decode-only%
(old, for contrast), backbone% (new), +ViT%, and the tax/escalated split. CPU-only (the ViT
itself is NOT run; only the processor counts tokens), reads existing labels + router_margin.pkl.
"""
import argparse, json, glob, os, re, pickle, time, numpy as np
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_7B = "/data/dan/weights/MedVLThinker-7B-RL_m23k"   # processor only (tokenizer + image proc)
ROOT     = "/data/dan/dataset/MedVLThinker-Eval"
EVAL     = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
SYS_NOTHINK = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28

# --- FLOPs constants (adjustable; ratio is dominated by N7/N32 ~ 0.23 and the token counts) ---
N7, N32, NVIT, MERGE = 7.6e9, 33.0e9, 0.675e9, 2     # Qwen2.5-VL-7B/32B backbones; shared ViT assumed
FACT = MERGE * MERGE                                  # 2x2 spatial merge -> patches = FACT * vision_tokens

ap = argparse.ArgumentParser()
ap.add_argument("--max_per_ds", type=int, default=0, help="0 = all; else cap per dataset for a quick look")
A = ap.parse_args()

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{cell}(?:_s\dof\d)?\.jsonl$"); d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip(): r = json.loads(l); d[m.group(1)][r["idx"]] = r
    return d
def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0]-v[1]) if len(v) >= 2 else 0.0
def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))

r7  = load_arm("ckpts/gate_7b_vllm", "nothink_norag")
r32 = load_arm("ckpts/gate_32b",     "think_norag")
R   = pickle.load(open("ckpts/router_margin.pkl", "rb")); gate, tau = R["gate"], R["tau"]
ds  = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]; data = ds[split]
proc = AutoProcessor.from_pretrained(MODEL_7B)
print(f"loaded processor + {len(data)} eval examples; counting prompt tokens (text+vision) per question...", flush=True)

def prompt_tokens(ex):
    """Exact P and pre-merge ViT patches for one question, mirroring run_7b_vllm.build()."""
    opts = parse_opts(ex["options"])
    q = ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
    im = [{"type": "image", "image": x, "max_pixels": HIGH_PX, "min_pixels": MIN_PX}
          for x in (ex.get("images") or [])]
    msgs = [{"role": "system", "content": SYS_NOTHINK},
            {"role": "user", "content": im + [{"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    out = proc(text=[text], images=imgs if imgs else None, return_tensors="pt")
    P = int(out["input_ids"].shape[1])
    patches = int(out["image_grid_thw"].prod(dim=-1).sum().item()) if "image_grid_thw" in out else 0
    return P, patches

rows = {"per_ds": [], "all": defaultdict(list)}
t0 = time.time()
for name in EVAL:
    if name not in r7 or name not in r32: continue
    idx = sorted(set(r7[name]) & set(r32[name]))
    if A.max_per_ds: idx = idx[:A.max_per_ds]
    P = np.zeros(len(idx)); patches = np.zeros(len(idx)); skip = 0
    G7  = np.array([float(r7[name][i].get("gen_tokens") or 0)  for i in idx])
    G32 = np.array([float(r32[name][i].get("gen_tokens") or 0) for i in idx])
    for j, i in enumerate(idx):
        try: P[j], patches[j] = prompt_tokens(data[i])
        except Exception as e:
            skip += 1
            if skip <= 3: print(f"   [{name}] token-count skip idx={i}: {e}", flush=True)
        if (j+1) % 500 == 0: print(f"   [{name}] {j+1}/{len(idx)} ({(time.time()-t0):.0f}s)", flush=True)
    mg  = np.array([[margin(r7[name][i])] for i in idx], dtype=np.float32)
    esc = gate.predict_proba(mg)[:, 1] < tau

    def ratios(include_vit):
        vit  = (2*NVIT*patches) if include_vit else np.zeros_like(P)
        run7  = 2*N7 *(P + G7)  + vit
        run32 = 2*N32*(P + G32) + vit
        base  = run32.sum()
        return ((run7.sum() + run32[esc].sum())/base, run7.sum()/base, run32[esc].sum()/base)
    bb, tax, et = ratios(False)
    wv, _, _    = ratios(True)
    old = ((N7*G7).sum() + (N32*G32[esc]).sum()) / (N32*G32).sum()    # decode-only proxy
    rows["per_ds"].append((name, len(idx), esc.mean(), P.mean(), (patches/FACT).mean(),
                           G7.mean(), G32.mean(), old, bb, tax, et, wv))
    # accumulate for FLOPs-weighted aggregate
    rows["all"]["run7"].append(2*N7*(P+G7)); rows["all"]["run32"].append(2*N32*(P+G32))
    rows["all"]["run7v"].append(2*N7*(P+G7)+2*NVIT*patches); rows["all"]["run32v"].append(2*N32*(P+G32)+2*NVIT*patches)
    rows["all"]["esc"].append(esc); rows["all"]["g7"].append(G7); rows["all"]["g32"].append(G32)
    print(f">> {name} done (skipped {skip})", flush=True)

print("\n" + "=" * 124)
print(f"PREFILL-INCLUSIVE COST   frozen margin gate (tau={tau:.3f})   N7={N7/1e9:g}B N32={N32/1e9:g}B ViT={NVIT/1e9:g}B(shared)")
print("=" * 124)
print(f"    {'dataset':<10}{'N':>6}{'32B-call%':>10}{'meanP':>8}{'meanVis':>8}{'meanG7':>8}{'meanG32':>9}"
      f" | {'decode%(old)':>13}{'backbone%':>11}{'+ViT%':>8}   (backbone = 7B-tax + 32B-esc)")
for (name, n, e, mp, mv, g7, g32, old, bb, tax, et, wv) in rows["per_ds"]:
    print(f"    {name:<10}{n:>6}{e*100:>9.0f}%{mp:>8.0f}{mv:>8.0f}{g7:>8.1f}{g32:>9.1f}"
          f" | {old*100:>12.0f}%{bb*100:>10.0f}%{wv*100:>7.0f}%   ({tax*100:.0f}% + {et*100:.0f}%)")

R7 = np.concatenate(rows["all"]["run7"]); R32 = np.concatenate(rows["all"]["run32"])
R7v = np.concatenate(rows["all"]["run7v"]); R32v = np.concatenate(rows["all"]["run32v"])
E = np.concatenate(rows["all"]["esc"]); G7a = np.concatenate(rows["all"]["g7"]); G32a = np.concatenate(rows["all"]["g32"])
agg_bb  = (R7.sum()  + R32[E].sum())  / R32.sum()
agg_v   = (R7v.sum() + R32v[E].sum()) / R32v.sum()
agg_old = ((N7*G7a).sum() + (N32*G32a[E].sum() if E.any() else 0)) / (N32*G32a).sum()
mean_ds_bb = np.mean([r[8] for r in rows["per_ds"]])
print("-" * 124)
print(f"    AVG over datasets (unweighted): backbone {mean_ds_bb*100:.0f}% of always-32B compute")
print(f"    GLOBAL (FLOPs-weighted):        decode-only(old) {agg_old*100:.0f}%  |  backbone {agg_bb*100:.0f}%  |  +ViT {agg_v*100:.0f}%")
print(f"    overall 32B-call rate: {E.mean()*100:.0f}%  (mechanism-agnostic headline; no FLOPs assumptions)")
print("\nREAD: the robust claim is the CALL RATE (never invokes 32B on ~{:.0f}% of questions). The".format((1-E.mean())*100))
print("backbone% is the honest prefill-inclusive compute (higher than the old decode-only proxy because")
print("the 7B's vision-heavy prefill is paid on every question). +ViT% adds the vision encoder. Quote the")
print("call rate as headline and backbone% as the compute figure; the old decode-only number was optimistic.")
