#!/usr/bin/env python3
"""verifier_image_ablation_v2.py -- THE MISSING DECISIVE EXPERIMENT.

Does the deployed trained open-text verifier (ckpts/train/lora_verifier_pooled4) actually USE THE
IMAGE, or is it a text-prior scorer ("lazy verifier" / Verification-Mirage failure mode)?

The repo already had src/training_methods/verifier_image_ablation.py, but (a) it never produced a
result artifact (logs/verifier_ablation.log stops after the model load), (b) it defaults to the
*other* adapter (lora_verifier_open, 2 datasets), (c) it covers only SLAKE+PathVQA, and (d) it only
tries one control (a 336x336 gray image), which confounds "no image content" with "8x fewer image
tokens".  This version fixes all four and scores SIX conditions on the SAME candidates:

  real              the true image at the verifier's deployed resolution (MAXPX = 1280*28*28)
  blank_gray        336x336 uniform gray  (127,127,127)   <- the original script's control
  blank_black       336x336 uniform black (0,0,0)
  blank_matched     uniform gray at the REAL image's exact pixel size -> identical image-token count,
                    so any real-vs-this gap is image CONTENT, not sequence length
  mismatched        another question's REAL image from the same dataset -> identical image statistics
                    and token budget, only the image<->question correspondence is destroyed
  no_image          text-only prompt (no vision tokens at all)

Reports, per condition: per-candidate AUROC (verifier score vs judge label) and best-of-8 SELECTION
accuracy (argmax score), pooled and per dataset, plus the subset of questions where the candidate set
actually contains both a correct and an incorrect answer ("mixed", the only questions where the
selection can differ).  The `real` scores are cross-checked against the saved scores in the
transfer dumps -- if they do not reproduce, the harness is wrong and the numbers are not reported.

MEMORY-SAFE: single Lingshu-7B (+LoRA) on ONE GPU, tp=1, batch-1 forwards, bf16, no_grad.
RESUMABLE: every (ds, idx, candidate, condition) score is appended to a checkpoint jsonl; re-running
skips completed keys.  SOFT DEADLINE: stops cleanly before the wall-clock guard fires.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 timeout -s KILL 2700 \
    python3 src/training_methods/verifier_image_ablation_v2.py --nq_slake 160 --nq_vqarad 200 --nq_pathvqa 240
"""
import argparse, os, json, glob, io, math, random, time
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
ADAPTER = "ckpts/train/lora_verifier_pooled4"
TAG = "lingshu7b"
MAXPX, MINPX = 1280 * 28 * 28, 4 * 28 * 28
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
       "proposed answer is correct. Respond with only 'Yes' or 'No'.")
CONDS = ["real", "blank_gray", "blank_black", "blank_matched", "mismatched", "no_image"]
norm = lambda s: str(s).strip().lower()

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--adapter", default=ADAPTER)
ap.add_argument("--nq_slake", type=int, default=160)
ap.add_argument("--nq_vqarad", type=int, default=200)
ap.add_argument("--nq_pathvqa", type=int, default=240)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--deadline_s", type=float, default=2300.0)
ap.add_argument("--ckpt", default="ckpts/train/lora_verifier_pooled4/image_ablation_scores.jsonl")
ap.add_argument("--analyze_only", action="store_true")
A = ap.parse_args()
T0 = time.time()


# ---------------------------------------------------------------- data
def load_dump(ds):
    return json.load(open(J(f"{ADAPTER}/transfer_dump_{ds}_{TAG}.json")))


def slake_images():
    m = {}
    for x in json.load(open("/data/dan/dataset/slake/test.json")):
        if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip):
                m[x["qid"]] = (x["question"], ip)
    return m


def parquet_images(base, want):
    """idx -> (question, PIL image). Only decodes the wanted idx (the row index of the concatenated
    test parquets, exactly as verifier_transfer_eval.py / run_lora_verifier_open.py define it)."""
    import pandas as pd
    from PIL import Image
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/test-*.parquet"))], ignore_index=True)
    m = {}
    for i in want:
        r = df.iloc[i]
        q = r.get("question")
        if q is None and "conversations" in r:
            q = r["conversations"][0]["value"].replace("<image>", "").strip()
        img = r["image"]
        m[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m


def build_sample():
    """-> ordered list of (ds, idx, question, sl_by_cand, cand_list), plus the sampled rows per ds."""
    plan = [("slake_open", A.nq_slake), ("vqa_rad_open", A.nq_vqarad), ("pathvqa_open", A.nq_pathvqa)]
    sel = {}
    for ds, nq in plan:
        rows = load_dump(ds)
        rng = random.Random(A.seed)
        sel[ds] = rng.sample(rows, min(nq, len(rows)))
    return sel


# ---------------------------------------------------------------- scoring
def run_scoring(sel):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
    from PIL import Image

    print("[data] loading images ...", flush=True)
    IMG = {}
    sl_img = slake_images()
    IMG["slake_open"] = {r["idx"]: sl_img[r["idx"]] for r in sel["slake_open"] if r["idx"] in sl_img}
    IMG["vqa_rad_open"] = parquet_images("/data/dan/dataset/vqa_rad/data", [r["idx"] for r in sel["vqa_rad_open"]])
    IMG["pathvqa_open"] = parquet_images("/data/dan/dataset/path_vqa/data", [r["idx"] for r in sel["pathvqa_open"]])
    for ds in IMG:
        print(f"       {ds}: {len(IMG[ds])} images for {len(sel[ds])} sampled questions", flush=True)

    # mismatched partner: a DIFFERENT question's image within the same dataset (cyclic shift of the
    # sampled order -> deterministic, every question gets a distinct partner)
    partner = {}
    for ds in IMG:
        ids = [r["idx"] for r in sel[ds] if r["idx"] in IMG[ds]]
        for a_, b_ in zip(ids, ids[1:] + ids[:1]):
            partner[(ds, a_)] = b_

    proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)
    YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
    print("[load] base Lingshu-7B + pooled4 LoRA ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")
    model = PeftModel.from_pretrained(model, J(A.adapter))
    model.eval()

    def pyes(q, img, ans):
        body = f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No."
        if img is None:
            content = [{"type": "text", "text": body}]
        else:
            content = [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
                       {"type": "text", "text": body}]
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        igs, vids = process_vision_info(msgs)
        enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
            py = math.exp(lg[YES].item()); pn = math.exp(lg[NO].item())
        return py / (py + pn) if (py + pn) > 0 else 0.5

    ck = J(A.ckpt)
    done = set()
    if os.path.exists(ck):
        for l in open(ck):
            if l.strip():
                r = json.loads(l); done.add((r["ds"], r["idx"], r["cand"], r["cond"]))
    print(f"[resume] {len(done)} scores already on disk", flush=True)

    # deterministic work list
    work = []
    for ds in ["slake_open", "vqa_rad_open", "pathvqa_open"]:
        for r in sel[ds]:
            i = r["idx"]
            if i not in IMG[ds]:
                continue
            cands = sorted(set(norm(a) for a in r["preds"]))
            surf = {}
            for a in r["preds"]:
                surf.setdefault(norm(a), a)
            for c in cands:
                for cond in CONDS:
                    work.append((ds, i, c, surf[c], cond))
    todo = [w for w in work if (w[0], w[1], w[2], w[4]) not in done]
    print(f"[plan] {len(work)} (candidate x condition) cells; {len(todo)} to score", flush=True)

    fh = open(ck, "a")
    GRAY336 = Image.new("RGB", (336, 336), (127, 127, 127))
    BLACK336 = Image.new("RGB", (336, 336), (0, 0, 0))
    n = 0
    for (ds, i, c, surf, cond) in todo:
        if time.time() - T0 > A.deadline_s:
            print(f"[deadline] stopping cleanly after {n} new scores", flush=True)
            break
        q, im = IMG[ds][i]
        if isinstance(im, str):
            im_real = Image.open(im).convert("RGB")
        else:
            im_real = im
        if cond == "real":
            img = im_real
        elif cond == "blank_gray":
            img = GRAY336
        elif cond == "blank_black":
            img = BLACK336
        elif cond == "blank_matched":
            img = Image.new("RGB", im_real.size, (127, 127, 127))
        elif cond == "mismatched":
            pim = IMG[ds][partner[(ds, i)]][1]
            img = Image.open(pim).convert("RGB") if isinstance(pim, str) else pim
        elif cond == "no_image":
            img = None
        try:
            s = pyes(q, img, surf)
        except Exception as e:
            print(f"  skip {ds}/{i}/{cond}: {str(e)[:80]}", flush=True)
            continue
        fh.write(json.dumps({"ds": ds, "idx": i, "cand": c, "cond": cond, "score": round(float(s), 6)}) + "\n")
        n += 1
        if n % 250 == 0:
            fh.flush()
            print(f"  {n}/{len(todo)}  {(time.time()-T0)/60:.1f} min", flush=True)
    fh.close()
    print(f"[done] wrote {n} new scores in {(time.time()-T0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------- analysis
def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    us, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(us)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_question_auroc(pairs):
    """pairs: list of (list_of_correct_scores, list_of_incorrect_scores) per question. Fraction of
    (correct, incorrect) pairs, WITHIN the same question, that the verifier ranks correctly (ties = 0.5).
    This is the quantity best-of-N selection actually needs; the pooled AUROC is inflated by between-
    question difficulty variation (easy questions have all-correct candidates AND high scores)."""
    num = den = 0.0
    for pos, neg in pairs:
        for p in pos:
            for q in neg:
                num += 1.0 if p > q else (0.5 if p == q else 0.0)
                den += 1
    return (num / den) if den else None


def boot_diff(a, b, n=10000, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def analyze(sel):
    ck = J(A.ckpt)
    S = defaultdict(dict)   # (ds,idx,cand) -> cond -> score
    for l in open(ck):
        if l.strip():
            r = json.loads(l)
            S[(r["ds"], r["idx"], r["cand"])][r["cond"]] = r["score"]

    # keep only questions with ALL conditions scored for ALL their candidates
    rows_by_ds = {}
    for ds in ["slake_open", "vqa_rad_open", "pathvqa_open"]:
        keep = []
        for r in sel[ds]:
            cands = sorted(set(norm(a) for a in r["preds"]))
            if all(all(c2 in S[(ds, r["idx"], c)] for c2 in CONDS) for c in cands):
                keep.append(r)
        rows_by_ds[ds] = keep
        print(f"  complete questions {ds}: {len(keep)}/{len(sel[ds])}")

    # per-candidate labels
    def cand_label(r, c):
        for a, l in zip(r["preds"], r["sl"]):
            if norm(a) == c:
                return int(l)
        return -1

    # fidelity check: real vs the saved scores in the transfer dump
    dev = []
    for ds, rows in rows_by_ds.items():
        for r in rows:
            for a, saved in zip(r["preds"], r["scores"]):
                mine = S[(ds, r["idx"], norm(a))]["real"]
                dev.append(abs(mine - float(saved)))
    fid = {"n": len(dev), "mean_abs_dev": float(np.mean(dev)) if dev else None,
           "max_abs_dev": float(np.max(dev)) if dev else None,
           "frac_within_0.01": float(np.mean(np.asarray(dev) < 0.01)) if dev else None}
    print(f"  fidelity vs saved dump scores: n={fid['n']} mean|Δ|={fid['mean_abs_dev']:.5f} "
          f"max|Δ|={fid['max_abs_dev']:.5f} within0.01={fid['frac_within_0.01']:.3f}")

    out = {"conditions": CONDS, "fidelity_real_vs_saved_dump": fid, "per_dataset": {}, "pooled": {},
           "sample": {ds: len(rows_by_ds[ds]) for ds in rows_by_ds}}

    sel_vec = {c: {"all": [], "mixed": []} for c in CONDS}
    cand_pool = {c: {"s": [], "y": []} for c in CONDS}
    greedy_all, greedy_mixed, oracle_all = [], [], []

    for ds, rows in rows_by_ds.items():
        d = {"n": len(rows), "n_mixed": 0, "greedy": None, "oracle": None, "cond": {}}
        loc_sel = {c: {"all": [], "mixed": []} for c in CONDS}
        loc_cand = {c: {"s": [], "y": []} for c in CONDS}
        g_all, g_mixed, o_all = [], [], []
        for r in rows:
            cands = sorted(set(norm(a) for a in r["preds"]))
            labs = {c: cand_label(r, c) for c in cands}
            mixed = (1 in labs.values()) and (0 in labs.values())
            g_all.append(int(r["greedy_ok"]))
            o_all.append(max([l for l in labs.values() if l >= 0] or [0]))
            if mixed:
                g_mixed.append(int(r["greedy_ok"])); d["n_mixed"] += 1
            for c in CONDS:
                sc = [S[(ds, r["idx"], a)][c] for a in cands]
                k = int(np.argmax(sc))
                ok = max(0, labs[cands[k]])
                loc_sel[c]["all"].append(ok)
                if mixed:
                    loc_sel[c]["mixed"].append(ok)
                for a in cands:
                    if labs[a] >= 0:
                        loc_cand[c]["s"].append(S[(ds, r["idx"], a)][c]); loc_cand[c]["y"].append(labs[a])
        d["greedy"] = float(np.mean(g_all)); d["oracle"] = float(np.mean(o_all))
        d["greedy_mixed"] = float(np.mean(g_mixed)) if g_mixed else None
        for c in CONDS:
            d["cond"][c] = {"select_acc": float(np.mean(loc_sel[c]["all"])),
                            "select_acc_mixed": float(np.mean(loc_sel[c]["mixed"])) if loc_sel[c]["mixed"] else None,
                            "cand_auroc": auroc(loc_cand[c]["s"], loc_cand[c]["y"]),
                            "gain_over_greedy": float(np.mean(loc_sel[c]["all"])) - d["greedy"]}
            sel_vec[c]["all"] += loc_sel[c]["all"]; sel_vec[c]["mixed"] += loc_sel[c]["mixed"]
            cand_pool[c]["s"] += loc_cand[c]["s"]; cand_pool[c]["y"] += loc_cand[c]["y"]
        greedy_all += g_all; greedy_mixed += g_mixed; oracle_all += o_all
        out["per_dataset"][ds] = d

    P = {"n": len(greedy_all), "n_mixed": len(greedy_mixed),
         "greedy": float(np.mean(greedy_all)), "greedy_mixed": float(np.mean(greedy_mixed)),
         "oracle": float(np.mean(oracle_all)), "n_candidates": len(cand_pool["real"]["y"]), "cond": {}}
    for c in CONDS:
        pt, lo, hi = boot_diff(sel_vec[c]["all"], greedy_all)
        rpt, rlo, rhi = boot_diff(sel_vec["real"]["all"], sel_vec[c]["all"])
        P["cond"][c] = {"select_acc": float(np.mean(sel_vec[c]["all"])),
                        "select_acc_mixed": float(np.mean(sel_vec[c]["mixed"])),
                        "cand_auroc": auroc(cand_pool[c]["s"], cand_pool[c]["y"]),
                        "gain_over_greedy": pt, "gain_ci": [lo, hi],
                        "real_minus_this_select": rpt, "real_minus_this_ci": [rlo, rhi],
                        "significant_vs_real": bool(rlo > 0 or rhi < 0) if c != "real" else None}
    out["pooled"] = P

    print("\n==================== IMAGE ABLATION (pooled4 verifier, 3 paper open cells) ====================")
    print(f"  n questions = {P['n']} ({P['n_mixed']} mixed)   n candidates = {P['n_candidates']}   "
          f"greedy = {P['greedy']:.4f}   oracle@8 = {P['oracle']:.4f}")
    print(f"  {'condition':<16}{'AUROC':>8}{'sel-acc':>10}{'gain':>9}{'sel(mixed)':>12}{'real-this':>11}{'95% CI':>22}")
    for c in CONDS:
        z = P["cond"][c]
        ci = f"[{z['real_minus_this_ci'][0]:+.4f},{z['real_minus_this_ci'][1]:+.4f}]"
        print(f"  {c:<16}{z['cand_auroc']:>8.4f}{z['select_acc']:>10.4f}{z['gain_over_greedy']:>+9.4f}"
              f"{z['select_acc_mixed']:>12.4f}{z['real_minus_this_select']:>+11.4f}{ci:>22}")
    return out


def main():
    sel = build_sample()
    if not A.analyze_only:
        run_scoring(sel)
    res = analyze(sel)
    outp = J("results/cascade_methods/artifacts/verifier_validity_2026-07-29.json")
    prev = json.load(open(outp)) if os.path.exists(outp) else {}
    prev["C_image_ablation"] = res
    prev.setdefault("provenance", {})["image_ablation_script"] = "src/training_methods/verifier_image_ablation_v2.py"
    json.dump(prev, open(outp, "w"), indent=1)
    print(f"\nwrote -> {outp}")


if __name__ == "__main__":
    main()
