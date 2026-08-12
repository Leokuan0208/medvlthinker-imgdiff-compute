#!/usr/bin/env python3
"""vram_levers_quant_acc.py -- ATTACK 4 part 2: the ACCURACY COST of quantising the CHEAP side.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
src/cascade/measure_vram_levers.py scenario Q measures what int8/int4 weight-only quantisation of
Lingshu-7B does to VRAM.  This script measures what it does to ACCURACY, on the two halves of the
unified pipeline separately:

  stage `mcq`    the 5 multiple-choice / closed reporting cells (62.5% of the macro weight),
                 through MedEvalKit's OWN dataset classes, prompts and cal_metrics.
  stage `verif`  the open-text half (37.5% of the macro weight) where the deployed decision is made
                 by the LoRA verifier: re-score the FROZEN 8-candidate pools with a quantised
                 verifier and recompute sel_eff with THE frozen metric
                 (src/training_methods/genframe_data.sel_eff).  No judge is involved: the
                 per-candidate correctness labels `sl` are already in the transfer dumps.

*** ONLY PAIRED DELTAS ARE REPORTED. ***  The absolute accuracies produced here are NOT comparable
to the published cells: this is HF transformers at batch 1, while every published MCQ cell was
produced by MedEvalKit under vLLM.  The bf16 control arm is run through THIS driver, on THESE items,
in THIS process shape, so quant-minus-bf16 differences out the serving stack entirely.  That is the
whole design; do not lift an absolute number out of this file.

*** THE STANDING FLOP CAVEAT, restated so it is never conflated. ***  int8/int4 WEIGHT-ONLY
quantisation performs ZERO fewer multiply-accumulates on sm80 (A100): the weight is dequantised (or
fed to a tinygemm kernel that accumulates in bf16) and the same bf16 GEMM runs.  It is a MEMORY
lever, and possibly a bandwidth/latency lever.  It is NOT a FLOP lever.  FLOPs are unchanged BY
CONSTRUCTION and are reported separately from memory, never added together.

REUSE, NOT REDO.  The model wrapper is `i8b_cheapleg_eval.HFVLM` (another round's driver), imported
and NEVER modified -- it already reproduces MedEvalKit's Qwen2.5-VL prompt construction verbatim.
It is driven at batch_size=1 here, which (a) matches the batch size of every VRAM row in
artifacts/vram_testtime_2026-08-11.json and artifacts/_vram_levers_parts/, and (b) sidesteps the
multi-image batch-collation failure that damaged that round's own large-batch PMC-VQA arm.

GPU ETIQUETTE: waits for free VRAM, never kills another process.  Resumable per arm+cell (JSONL).

  python3 src/cascade_methods/vram_levers_quant_acc.py --stage mcq   --arms bf16,int8wo,int4wo
  python3 src/cascade_methods/vram_levers_quant_acc.py --stage verif --arms bf16,int8wo,int4wo
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEK = os.path.join(REPO, "MedEvalKit")
ART = os.path.join(REPO, "results/cascade_methods/artifacts/_vram_levers_parts")
CKPT = os.path.join(REPO, "ckpts/vram_levers_quant")
os.makedirs(ART, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

L7B = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/"
       "snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9")
VERIFIER = os.path.join(REPO, "ckpts/train/lora_verifier_disjoint")

# the 5 MCQ/closed reporting cells: MedEvalKit dataset -> (cell name, row filter)
MCQ_CELLS = [
    ("PMC_VQA", "PMC_VQA", None),
    ("SLAKE", "SLAKE_closed", "SLAKE_CLOSED"),
    ("VQA_RAD", "VQA_RAD_closed", "YESNO"),
    ("PATH_VQA", "PATH_VQA_closed", "YESNO"),
    ("MedXpertQA-MM", "MedXpertQA-MM", None),
]
# verifier prompt -- VERBATIM from src/training_methods/cheapleg_score_open.py:31-33
V_MAXPX, V_MINPX = 1280 * 28 * 28, 4 * 28 * 28
V_SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide "
         "whether the proposed answer is correct. Respond with only 'Yes' or 'No'.")

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["mcq", "verif"])
ap.add_argument("--arms", default="bf16,int8wo,int4wo")
ap.add_argument("--n_mcq", type=int, default=250, help="items per MCQ cell (seeded subsample)")
ap.add_argument("--n_verif", type=int, default=150, help="questions per open cell (seeded subsample)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max_new_tokens", type=int, default=64)
ap.add_argument("--wait_mb", type=int, default=26000)
ap.add_argument("--wait_timeout_s", type=int, default=25200)
A = ap.parse_args()

os.environ.setdefault("HF_HOME", "/data/dan/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# MedEvalKit reads its whole configuration from the environment at import time.
for k, v in dict(REASONING="False", datasets_path="hf", use_llm_judge="False",
                 judge_model_type="openai", judge_model="None", api_key="None",
                 base_url="None", use_vllm="False", max_image_num="6",
                 test_times="1", seed="42").items():
    os.environ.setdefault(k, v)


# ------------------------------------------------------------------ GPU etiquette (never kill)
def free_mb_per_gpu():
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True).strip().splitlines()
    return [int(t) - int(u) for u, t in (l.split(",") for l in out)]


def pick_gpu(need_mb, timeout_s):
    t0 = time.time()
    while True:
        free = free_mb_per_gpu()
        best = max(range(len(free)), key=lambda i: free[i])
        if free[best] >= need_mb:
            print(f"[wait] GPU {best} has {free[best]} MB free (need {need_mb}) -- proceeding",
                  flush=True)
            return best
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"no GPU with {need_mb} MB free after {timeout_s}s (free={free})")
        print(f"[wait] free={free}, need {need_mb}; sleeping 120 s", flush=True)
        time.sleep(120)


DEV = pick_gpu(A.wait_mb, A.wait_timeout_s)
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEV)

import numpy as np                                                            # noqa: E402
import torch                                                                  # noqa: E402

# NUMERICS PIN -- stated because the project has measured TF32 worth -0.0089/+0.024 on this kind of
# comparison.  Both arms of every contrast run under the identical setting, inside one process.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
NUMERICS = dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                torch=torch.__version__, dtype="bfloat16", attn_implementation="sdpa",
                batch_size=1, tensor_parallel=1, greedy=True, seed=A.seed,
                omp_num_threads=os.environ.get("OMP_NUM_THREADS", "unset"))

sys.path.insert(0, os.path.join(REPO, "src/cascade_methods"))
sys.path.insert(0, os.path.join(REPO, "src/training_methods"))


def quantise(model, scheme):
    """Weight-only quantisation of the LANGUAGE MODEL ONLY (the vision tower stays bf16).

    The tower is ~0.68 B of 8.29 B parameters and int4 group-wise quantisation of a ViT is a
    separate, separately contentious intervention -- recorded here, not silently assumed.
    """
    if scheme == "bf16":
        return dict(scheme="bf16", applied_to=None, library=None)
    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
    import torchao
    cfg = Int8WeightOnlyConfig() if scheme == "int8wo" else Int4WeightOnlyConfig()
    target = model.model.language_model if hasattr(model.model, "language_model") else model.model
    n_lin = sum(1 for m in target.modules() if m.__class__.__name__ == "Linear")
    quantize_(target, cfg)
    torch.cuda.synchronize()
    return dict(scheme=scheme, applied_to="language_model only (vision tower left bf16)",
                library=f"torchao {torchao.__version__}", n_linear_modules_targeted=int(n_lin),
                flop_note="ZERO MAC reduction on sm80; memory lever only, never a FLOP lever")


def _as_pil(x):
    """MedEvalKit hands some benchmarks a PIL image and others a PATH STRING relative to
    MedEvalKit/ (PMC-VQA is the latter).  transformers' fast image processor rejects the string
    with `Could not make a flat list of images`, the whole batch is caught by the driver's error
    guard, and the cell silently scores near zero.  That defect is visible in
    ckpts/i8b_cheapleg/base7b/PMC_VQA/metrics.json (acc 0.1323 against a published 0.5427), so it
    is repaired HERE rather than in that round's file, which is imported and never modified.
    """
    from PIL import Image
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    return x


def _patched_build(self, messages):
    """i8b_cheapleg_eval.HFVLM._build with image PATHS resolved to PIL.  Everything else -- the
    prompt text, the system message, the chat template, the ordering -- is that method verbatim."""
    prompt = messages["prompt"]
    system = messages.get("system")
    imgs = []
    if "image" in messages:
        imgs = [_as_pil(messages["image"])]
    elif "images" in messages:
        imgs = [_as_pil(i) for i in messages["images"]]
    content = [{"type": "image", "image": im} for im in imgs]
    content.append({"type": "text", "text": prompt})
    conv = []
    if system:
        conv.append({"role": "system", "content": system})
    conv.append({"role": "user", "content": content})
    text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    return text, imgs


def load_arm(scheme, adapter=None):
    import i8b_cheapleg_eval as drv          # another round's driver -- imported, NEVER modified
    t0 = time.time()
    m = drv.HFVLM(L7B, "qwen2_5_vl", max_new_tokens=A.max_new_tokens, batch_size=1,
                  device="cuda:0")
    # bind the repaired builder to THIS instance only; drv's module state is untouched
    m._build = _patched_build.__get__(m, type(m))
    q = quantise(m.model, scheme)
    if adapter:
        from peft import PeftModel
        m.model = PeftModel.from_pretrained(m.model, adapter)
        m.model.eval()
    torch.cuda.synchronize()
    info = dict(quantisation=q, load_s=round(time.time() - t0, 1),
                weights_resident_gib=round(torch.cuda.memory_allocated() / 1024 ** 3, 4),
                adapter=adapter, model_path=L7B)
    print(f"[load] {scheme} adapter={bool(adapter)} resident={info['weights_resident_gib']} GiB "
          f"({info['load_s']} s)", flush=True)
    return m, drv, info


# =============================================================================== stage: mcq
def subsample(n_total, k, seed, tag):
    rng = random.Random(f"{seed}|{tag}")
    idx = list(range(n_total))
    rng.shuffle(idx)
    return sorted(idx[:k])


def row_keep(sample, filt):
    if filt is None:
        return True
    if filt == "SLAKE_CLOSED":
        return str(sample.get("answer_type", "")).upper() == "CLOSED"
    if filt == "YESNO":
        return str(sample.get("answer", "")).strip().lower() in ("yes", "no")
    return True


def stage_mcq(arms):
    sys.path.insert(0, MEK)
    cwd0 = os.getcwd()
    os.chdir(MEK)
    from benchmarks import prepare_benchmark
    out = {}
    for scheme in arms:
        m, drv, info = load_arm(scheme)
        out[scheme] = dict(_load=info, cells={})
        for ds, cell, filt in MCQ_CELLS:
            odir = os.path.join(CKPT, scheme, cell)
            os.makedirs(odir, exist_ok=True)
            mp = os.path.join(odir, "metrics.json")
            if os.path.exists(mp):
                out[scheme]["cells"][cell] = json.load(open(mp))
                print(f"[skip done] {scheme} {cell}", flush=True)
                continue
            try:
                dataset = prepare_benchmark(m, ds, None, odir)
                dataset.load_data()
                allsamp = dataset.samples
                keep = [i for i in range(len(allsamp)) if row_keep(allsamp[i], filt)]
                pick = [keep[j] for j in subsample(len(keep), min(A.n_mcq, len(keep)),
                                                  A.seed, cell)]
                # MedEvalKit's SLAKE / VQA_RAD / PATH_VQA cal_metrics divides by the OPEN-question
                # count, so a closed-only sample list raises ZeroDivisionError.  Append a few
                # complement rows purely to keep that denominator non-zero.  They are APPENDED, so
                # positions 0..len(pick)-1 are unchanged (the resume file stays valid), and they are
                # EXCLUDED from the reported accuracy.
                pad = []
                if filt:
                    keepset = set(keep)          # hoisted: rebuilding it per element is O(n^2)
                    rest = [i for i in range(len(allsamp)) if i not in keepset]
                    if rest:
                        pad = [rest[j] for j in subsample(len(rest), min(3, len(rest)), A.seed,
                                                          "pad|" + cell)]
                dataset.samples = [allsamp[i] for i in pick] + [allsamp[i] for i in pad]
                print(f"### {scheme} {cell}: {len(pick)} of {len(keep)} eligible "
                      f"({len(allsamp)} raw) + {len(pad)} denominator-pad rows", flush=True)
                t0 = time.time()
                samples = drv.resumable_run(dataset, m, os.path.join(odir, "gen.jsonl"))
                metrics, scored = dataset.cal_metrics(samples)
                scored = scored[:len(pick)]                  # drop the pad rows before scoring
                ok = [int(bool(s.get("correct"))) for s in scored]
                rec = dict(cell=cell, medevalkit_dataset=ds, row_filter=filt,
                           n=len(ok), acc=round(float(np.mean(ok)), 6),
                           n_eligible=len(keep), n_raw=len(allsamp),
                           selected_row_indices=pick, denominator_pad_row_indices=pad,
                           per_item_ok=ok,
                           n_empty_response=sum(1 for s in scored
                                                if not str(s.get("response", "")).strip()),
                           mean_gen_tokens=round(float(np.mean(
                               [s.get("gen_toks", 0) or 0 for s in scored])), 2),
                           wall_s=round(time.time() - t0, 1),
                           medevalkit_metrics=metrics)
                json.dump(rec, open(mp, "w"), indent=1)
                out[scheme]["cells"][cell] = rec
                print(f"[{scheme} {cell}] n={rec['n']} acc={rec['acc']} "
                      f"({rec['wall_s']} s)", flush=True)
            except Exception as e:                                   # per-cell error guard
                traceback.print_exc()
                out[scheme]["cells"][cell] = dict(cell=cell, error=f"{type(e).__name__}: {e}")
            json.dump(out, open(os.path.join(ART, "QACC_mcq.json"), "w"), indent=1)
        del m
        torch.cuda.empty_cache()
    os.chdir(cwd0)
    return out


# ============================================================================= stage: verif
def stage_verif(arms):
    """Re-score the FROZEN 8-candidate pools with a quantised verifier; recompute THE metric."""
    import genframe_data as G
    from qwen_vl_utils import process_vision_info
    items_all = G.load_items()
    # deterministic subsample, per dataset, of the canonical pool
    by_ds = {}
    for i, it in enumerate(items_all):
        by_ds.setdefault(it["ds"], []).append(i)
    sel = []
    for ds in sorted(by_ds):
        idx = by_ds[ds]
        sel += [idx[j] for j in subsample(len(idx), min(A.n_verif, len(idx)), A.seed, "verif|" + ds)]
    sel = sorted(sel)
    items = [items_all[i] for i in sel]
    print(f"[verif] {len(items)} questions of {len(items_all)} "
          f"({ {d: sum(1 for it in items if it['ds'] == d) for d in by_ds} })", flush=True)

    imgs = {}
    for ds in by_ds:
        imgs[ds] = _imgs_for(ds)

    out = dict(_pool=dict(n_questions=len(items), n_pool_total=len(items_all),
                          selected_positions=sel,
                          per_ds={d: sum(1 for it in items if it["ds"] == d) for d in by_ds}))
    for scheme in arms:
        m, drv, info = load_arm(scheme, adapter=VERIFIER)
        proc = m.processor
        yes = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no = proc.tokenizer.encode("No", add_special_tokens=False)[0]
        path = os.path.join(CKPT, f"verif_{scheme}.jsonl")
        done = {}
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                    done[(r["ds"], r["idx"])] = r["scores"]
                except Exception:
                    pass
        fh = open(path, "a")
        t0 = time.time()
        for k, it in enumerate(items):
            key = (it["ds"], it["idx"])
            if key in done:
                continue
            try:
                im = imgs[it["ds"]][it["idx"]]
                q = im["question"]
                sc = []
                for cand in it["preds"]:
                    msgs = [{"role": "system", "content": V_SYS},
                            {"role": "user", "content": [
                                {"type": "image", "image": im["img"],
                                 "max_pixels": V_MAXPX, "min_pixels": V_MINPX},
                                {"type": "text", "text":
                                 f"Question: {q}\nProposed answer: {cand}\n"
                                 f"Is the proposed answer correct? Answer Yes or No."}]}]
                    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    ims, vids = process_vision_info(msgs)
                    enc = proc(text=[text], images=ims, videos=vids, return_tensors="pt",
                               padding=True).to("cuda")
                    with torch.no_grad():
                        lg = m.model(**enc).logits[0, -1].float()
                    p = torch.softmax(torch.stack([lg[yes], lg[no]]), dim=0)[0].item()
                    sc.append(round(float(p), 6))
                    del enc, lg
                done[key] = sc
                fh.write(json.dumps(dict(ds=it["ds"], idx=it["idx"], scores=sc)) + "\n")
                fh.flush()
            except Exception as e:                                    # per-item error guard
                print(f"  verif fail {key}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                torch.cuda.empty_cache()
            if (k + 1) % 50 == 0:
                print(f"  [{scheme}] {k+1}/{len(items)}  {time.time()-t0:.0f} s", flush=True)
        fh.close()
        scored = [it for it in items if (it["ds"], it["idx"]) in done]
        picks = [int(np.argmax(done[(it["ds"], it["idx"])])) for it in scored]
        res = G.sel_eff({(it["ds"], it["idx"]): done[(it["ds"], it["idx"])] for it in scored},
                        items=scored)
        out[scheme] = dict(
            _load=info, n_scored=len(scored), n_requested=len(items),
            sel_eff=res["sel_eff"], selected=res["acc"], oracle=res["oracle"],
            greedy=res["greedy"], n_recoverable=int(res["n_recoverable"]),
            per_ds={k2: dict(n=int(v["n"]), sel_eff=v["sel_eff"], acc=v["acc"],
                             oracle=v["oracle"]) for k2, v in res["per_ds"].items()},
            contested=dict(n=int(res["contested"]["n"]),
                           sel_eff=res["contested"]["sel_eff"]),
            picks=picks, got=[int(x) for x in res["got"]],
            wall_s=round(time.time() - t0, 1))
        print(f"[{scheme}] sel_eff={res['sel_eff']:.6f} on n={len(scored)}", flush=True)
        json.dump(out, open(os.path.join(ART, "QACC_verif.json"), "w"), indent=1)
        del m
        torch.cuda.empty_cache()
    return out


def _imgs_for(ds):
    """Images + questions, VERBATIM from src/training_methods/cheapleg_score_open.py:imgs_for."""
    import glob, io                                                            # noqa: E401
    import pandas as pd
    from PIL import Image
    m = {}
    if ds == "slake_open":
        for x in json.load(open("/data/dan/dataset/slake/test.json")):
            if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
                ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
                if os.path.exists(ip):
                    m[x["qid"]] = dict(img=ip, question=x["question"])
        return m
    base = {"vqa_rad_open": "/data/dan/dataset/vqa_rad/data",
            "pathvqa_open": "/data/dan/dataset/path_vqa/data"}[ds]
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(base + "/test-*.parquet"))],
                   ignore_index=True)
    for i, r in df.iterrows():
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            m[int(i)] = dict(img=Image.open(io.BytesIO(img["bytes"])).convert("RGB"),
                             question=str(r.get("question")))
    return m


def main():
    arms = [a.strip() for a in A.arms.split(",") if a.strip()]
    env = dict(date="2026-08-12", stage=A.stage, arms=arms, gpu_physical_index=DEV,
               numerics=NUMERICS, n_mcq=A.n_mcq, n_verif=A.n_verif, seed=A.seed,
               code="src/cascade_methods/vram_levers_quant_acc.py",
               framework="HuggingFace transformers (NEVER vLLM)",
               absolutes_are_not_comparable_to_published=(
                   "HF batch-1 driver; every published MCQ cell is MedEvalKit under vLLM. Only the "
                   "paired quant-minus-bf16 delta from THIS file is interpretable."))
    res = stage_mcq(arms) if A.stage == "mcq" else stage_verif(arms)
    p = os.path.join(ART, f"QACC_{A.stage}.json")
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.update(res)
    cur["_env"] = env
    json.dump(cur, open(p, "w"), indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
