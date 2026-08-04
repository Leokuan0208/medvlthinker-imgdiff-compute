#!/usr/bin/env python3
"""extract_generator_hidden.py -- cache the GENERATOR's own hidden states for every
(image, question, candidate-answer) triple, so a DISCRIMINATIVE HEAD can be trained on them
instead of prompting a language model for an opinion.

WHY. Every verifier tried in this project has been a generative LM emitting an opinion (prompted
or LoRA-tuned); the incumbent's score is P(Yes) read off the LM head. This script produces the raw
material for the alternative: a frozen forward pass of the *base* Lingshu-7B (the actual generator,
NO adapter -- an adapter would make the features a function of the incumbent) with the hidden
states saved, so that logistic / MLP / listwise / Bradley-Terry heads can be fit on CPU afterwards.

TWO PROMPT FORMATS (--mode), one forward pass each:
  grader     the verifier prompt VERBATIM from run_lora_verifier_disjoint.py (same SYS, same body,
             same max_pixels).  Readout position = last prompt token, i.e. exactly the position
             whose representation the LM head turns into Yes/No.  This is the CONTROLLED contrast:
             identical base model, identical prompt, identical training examples as the incumbent
             LoRA -- the ONLY thing that changes is the readout (trained head vs fine-tuned LM head).
  generator  the generator's OWN answering prompt (run_openvqa.py SYS) with the candidate placed as
             the assistant turn.  Readout = last answer token + mean over answer tokens.  This is
             the "last-answer-token probe" (arXiv:2410.02707): does the generator internally know
             the answer it produced is wrong?

WHAT IS SAVED per row, per layer L in --layers (indices into output_hidden_states; 0 = embeddings,
28 = final layer of Lingshu-7B/Qwen2.5-VL-7B):
  h_last[L]  hidden state at the readout position
  h_span[L]  mean hidden state over the span of interest (grader: all text tokens after the image;
             generator: the assistant answer tokens)
Plus, free from the same pass, the base model's zero-shot Yes/No logits at the grader readout
position (mode=grader only) -- this reproduces the "generator scoring its own candidates" control.

DISJOINTNESS. Rows carry the md5 of their DECODED RGB pixels; assert_disjoint() in the fitting
script checks eval-image / train-image hash intersection == 0 (project rule; contamination once
inflated a verifier 2.9x).

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
    src/training_methods/extract_generator_hidden.py --mode grader --split eval --out feats_hidden
"""
import argparse, os, json, glob, io, math, random, time, hashlib
import numpy as np, torch
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
TAG = "lingshu7b"
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28

# verbatim from run_lora_verifier_disjoint.py / verifier_transfer_eval.py
SYS_GRADER = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
              "proposed answer is correct. Respond with only 'Yes' or 'No'.")
# verbatim from src/labeling/run_openvqa.py (the prompt that GENERATED these candidates)
SYS_GEN = "You are an expert medical image analyst. Answer the question with a short, specific phrase. Do not explain."

FAMILY = {"slake_open_train": "slake", "pathvqa_open_train": "pathvqa",
          "vqa_rad_open_train": "vqa_rad", "kvasir_open": "kvasir", "radimagenet_open": "radimagenet"}
CONTAMINATED_QUOTA = {"slake": 894, "pathvqa": 4973, "vqa_rad": 522, "kvasir": 3975}
EVAL_DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]


def loadj(p):
    return {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}


def norm(s):
    return str(s).strip().lower()


def img_md5(im):
    """md5 of DECODED RGB pixels -- the project's contamination assertion currency."""
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    return hashlib.md5(im.convert("RGB").tobytes()).hexdigest()


# ---------------------------------------------------------------- image/question maps
def slake_imgs(split):
    m = {}
    for x in json.load(open(f"/data/dan/dataset/slake/{split}.json")):
        if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip):
                m[x["qid"]] = (x["question"], ip)
    return m


def parquet_imgs(base, split, want=None):
    import pandas as pd
    m = {}
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/{split}-*.parquet"))], ignore_index=True)
    for i, r in df.iterrows():
        if want is not None and int(i) not in want:
            continue
        q = r.get("question"); a = r.get("answer")
        if q is None and "conversations" in r:
            conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
        if str(a).strip().lower() in ("yes", "no"):
            continue
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            m[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m


def json_imgs(jp):
    return {r["idx"]: (r["question"], r["img_path"]) for r in json.load(open(jp)) if os.path.exists(r["img_path"])}


def imgs_for_eval(ds):
    if ds == "slake_open":
        return slake_imgs("test")
    base = "/data/dan/dataset/vqa_rad/data" if ds == "vqa_rad_open" else "/data/dan/dataset/path_vqa/data"
    return parquet_imgs(base, "test")


# ---------------------------------------------------------------- row construction
def judged_answers(ds):
    """{item_idx: {normalized_answer: judge_ok}} and {item_idx: [surface preds in pool order]}"""
    sc = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8.jsonl")
    exp = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.jsonl")
    jud = {k: v["judge_ok"] for k, v in loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.judge.jsonl").items()}
    aj = defaultdict(dict)
    for cid, r in exp.items():
        if cid in jud:
            oi = cid.split("#")[0]; oi = int(oi) if oi.lstrip("-").isdigit() else oi
            aj[oi][norm(r["modal_pred"])] = jud[cid]
    return sc, aj


def build_eval_rows():
    rows = []
    for ds in EVAL_DS:
        sc, aj = judged_answers(ds)
        IMG = imgs_for_eval(ds)
        for i in sc:
            if i not in aj or i not in IMG:
                continue
            q, img = IMG[i]
            preds = sc[i]["preds"]
            sl = [aj[i].get(norm(a)) for a in preds]
            if all(x is None for x in sl):
                continue
            seen = {}
            for a in preds:
                na = norm(a)
                if na in seen or na not in aj[i]:
                    continue
                seen[na] = 1
                rows.append({"split": "eval", "ds": ds, "idx": i, "na": na, "ans": a,
                             "y": int(aj[i][na]), "q": q, "img": img})
    return rows


def build_train_rows(level="L1", seed=0, max_train=10364, match_composition=True, full_groups=True):
    """Reproduces run_lora_verifier_disjoint.py's seed-0 matched draw EXACTLY (verified: the
    resulting pos_rate 0.19924739482825163 equals the value recorded in its train_config.json).
    With full_groups=True we additionally extract the REMAINING answers of every question the draw
    touched, so listwise / Bradley-Terry heads have complete within-question groups.  The QUESTION
    set is unchanged; only extra answers of already-selected questions are added."""
    SPL = json.load(open(os.path.join(ROOT, "results/cascade_methods/artifacts/verifier_disjoint_split.json")))
    assert SPL["disjointness_assertion"]["image_pixel_hash_intersection"] == 0
    DSETS = list(SPL["train"].keys())
    pref = "idx_" if level == "L1" else "strict_idx_"
    ALLOW = {ds: set(json.load(open(os.path.join(ROOT, "data/disjoint_split", f"{pref}{ds}.json")))) for ds in DSETS}

    IMG = {}
    for ds in DSETS:
        if ds == "slake_open_train":     IMG[ds] = slake_imgs("train")
        elif ds == "vqa_rad_open_train": IMG[ds] = parquet_imgs("/data/dan/dataset/vqa_rad/data", "train", ALLOW[ds])
        elif ds == "pathvqa_open_train": IMG[ds] = parquet_imgs("/data/dan/dataset/path_vqa/data", "train", ALLOW[ds])
        elif ds == "kvasir_open":        IMG[ds] = json_imgs("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json")
        elif ds == "radimagenet_open":   IMG[ds] = json_imgs("/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")
        IMG[ds] = {k: v for k, v in IMG[ds].items() if k in ALLOW[ds]}

    QREC = {}
    for ds in DSETS:
        sc, aj = judged_answers(ds)
        if not sc or not aj:
            continue
        for i in sc:
            if i not in ALLOW[ds] or i not in IMG[ds] or i not in aj:
                continue
            q, img = IMG[ds][i]
            QREC[(ds, i)] = {"q": q, "img": img, "preds": sc[i]["preds"], "slabels": aj[i]}

    rng = random.Random(seed)
    by_src = defaultdict(list)
    for k in sorted(QREC, key=lambda t: (t[0], str(t[1]))):
        r = QREC[k]
        for na, lab in r["slabels"].items():
            surf = next((a for a in r["preds"] if norm(a) == na), na)
            by_src[k[0]].append((k, na, surf, lab))
    draw = []
    if match_composition:
        for ds in DSETS:
            fam = FAMILY[ds]
            if fam == "radimagenet":
                continue
            pool = by_src.get(ds, []); rng.shuffle(pool)
            draw += pool[:CONTAMINATED_QUOTA[fam]]
        deficit = max_train - len(draw)
        if deficit > 0:
            pool = by_src.get("radimagenet_open", []); rng.shuffle(pool)
            draw += pool[:deficit]
    else:
        for ds in DSETS:
            draw += by_src.get(ds, [])
        rng.shuffle(draw); draw = draw[:max_train]
    in_draw = set((k, na) for (k, na, _, _) in draw)
    qset = set(k for (k, _, _, _) in draw)
    print(f"[train] matched draw={len(draw)} examples over {len(qset)} questions; "
          f"pos_rate={np.mean([d[3] for d in draw]):.17f}", flush=True)

    rows = []
    src = draw if not full_groups else [x for ds in DSETS for x in by_src.get(ds, []) if x[0] in qset]
    for (k, na, surf, lab) in src:
        ds, i = k; r = QREC[k]
        rows.append({"split": "train", "ds": ds, "idx": i, "na": na, "ans": surf, "y": int(lab),
                     "q": r["q"], "img": r["img"], "in_draw": int((k, na) in in_draw)})
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--mode", choices=["grader", "generator"], default="grader")
    ap.add_argument("--split", choices=["eval", "train"], required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21, 28])
    ap.add_argument("--out", default="feats_hidden")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify_memo", type=int, default=0,
                    help="assert the per-item memo reproduces the unmemoized processor call exactly")
    A = ap.parse_args()
    DEV = "cuda"
    outdir = os.path.join(ROOT, A.out); os.makedirs(outdir, exist_ok=True)
    stem = f"{A.mode}_{A.split}" + (f"_s{A.shard}of{A.nshard}" if A.nshard > 1 else "")

    print(f"[build] rows for split={A.split} ...", flush=True)
    rows = build_eval_rows() if A.split == "eval" else build_train_rows()
    rows.sort(key=lambda r: (r["ds"], str(r["idx"]), r["na"]))
    if A.limit:
        rows = rows[:A.limit]
    rows = rows[A.shard::A.nshard]
    print(f"[build] {len(rows)} rows (shard {A.shard}/{A.nshard})", flush=True)

    proc = AutoProcessor.from_pretrained(A.model_path)
    tok = proc.tokenizer
    YES = tok.encode("Yes", add_special_tokens=False)[0]
    NO = tok.encode("No", add_special_tokens=False)[0]
    IMG_TOK = tok.convert_tokens_to_ids("<|image_pad|>")
    print(f"loading base {A.model_path} (NO adapter) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
    model.eval()
    H = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    print(f"hidden={H}; layers={A.layers}", flush=True)

    # ---- per-ITEM memo. Rows are sorted by (ds, idx, answer), so an item's 3-8 candidates are
    # consecutive and share one image: resize + patchify + pixel-md5 are done ONCE per item instead
    # of once per candidate (~4x fewer image ops on eval, ~5x on train). The text still varies per
    # row. Verified byte-identical to the unmemoized path (see --verify_memo).
    MERGE = proc.image_processor.merge_size
    memo = {"key": None}

    def image_inputs(r):
        k = (r["ds"], r["idx"])
        if memo["key"] != k:
            msgs_img = [{"role": "user", "content": [{"type": "image", "image": r["img"],
                                                      "max_pixels": HIGH_PX, "min_pixels": MIN_PX}]}]
            igs, _ = process_vision_info(msgs_img)
            ip = proc.image_processor(images=igs, return_tensors="pt")
            memo.update({"key": k, "pv": ip["pixel_values"], "grid": ip["image_grid_thw"],
                         "md5": img_md5(r["img"])})
        return memo["pv"], memo["grid"], memo["md5"]

    def encode(text, r):
        pv, grid, md5 = image_inputs(r)
        npad = int(grid.prod().item()) // (MERGE ** 2)
        t = text.replace("<|image_pad|>", "<|placeholder|>" * npad).replace("<|placeholder|>", "<|image_pad|>")
        enc = proc.tokenizer([t], return_tensors="pt", padding=True)
        enc["pixel_values"] = pv; enc["image_grid_thw"] = grid
        return enc, md5

    n = len(rows)
    h_last = np.zeros((n, len(A.layers), H), dtype=np.float16)
    h_span = np.zeros((n, len(A.layers), H), dtype=np.float16)
    yesno = np.full((n, 2), np.nan, dtype=np.float32)
    meta, bad = [], 0
    t0 = time.time()
    for r_i, r in enumerate(rows):
        try:
            if A.mode == "grader":
                body = (f"Question: {r['q']}\nProposed answer: {r['ans']}\n"
                        f"Is the proposed answer correct? Answer Yes or No.")
                msgs = [{"role": "system", "content": SYS_GRADER},
                        {"role": "user", "content": [{"type": "image", "image": r["img"],
                                                      "max_pixels": HIGH_PX, "min_pixels": MIN_PX},
                                                     {"type": "text", "text": body}]}]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                n_ans_tok = 0
            else:
                msgs = [{"role": "system", "content": SYS_GEN},
                        {"role": "user", "content": [{"type": "image", "image": r["img"],
                                                      "max_pixels": HIGH_PX, "min_pixels": MIN_PX},
                                                     {"type": "text", "text": r["q"]}]}]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                n_ans_tok = len(tok.encode(r["ans"], add_special_tokens=False))
                text = text + r["ans"]
            enc, md5 = encode(text, r)
            if A.verify_memo:
                igs, vids = process_vision_info(msgs)
                ref = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True)
                assert torch.equal(ref["input_ids"], enc["input_ids"]), "memo: input_ids differ"
                assert torch.allclose(ref["pixel_values"], enc["pixel_values"]), "memo: pixels differ"
            enc = enc.to(DEV)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states
            ids = enc["input_ids"][0]
            if A.mode == "grader":
                # span = every text token after the image block (question + proposed answer + instruction)
                pos = (ids == IMG_TOK).nonzero()
                s0 = int(pos[-1].item()) + 1 if len(pos) else 0
                readout = ids.shape[0] - 1
                lg = out.logits[0, -1]
                yesno[r_i] = [float(lg[YES].item()), float(lg[NO].item())]
            else:
                # span = the assistant answer tokens; readout = last answer token
                s0 = max(0, ids.shape[0] - n_ans_tok)
                readout = ids.shape[0] - 1
            for li, L in enumerate(A.layers):
                h = hs[L][0]
                h_last[r_i, li] = h[readout].float().cpu().numpy().astype(np.float16)
                h_span[r_i, li] = h[s0:].float().mean(0).cpu().numpy().astype(np.float16)
            meta.append({"split": r["split"], "ds": r["ds"], "idx": r["idx"], "na": r["na"], "ans": r["ans"],
                         "y": r["y"], "in_draw": r.get("in_draw", 1),
                         "img_md5": md5, "n_tok": int(ids.shape[0])})
        except Exception as e:
            bad += 1
            meta.append({"split": r["split"], "ds": r["ds"], "idx": r["idx"], "na": r["na"], "ans": r["ans"],
                         "y": r["y"], "in_draw": r.get("in_draw", 1), "img_md5": None, "n_tok": -1,
                         "err": str(e)[:120]})
            print(f"  skip {r['ds']}/{r['idx']}: {str(e)[:100]}", flush=True)
        if (r_i + 1) % 250 == 0:
            el = time.time() - t0
            print(f"  {r_i+1}/{n}  {el/60:.1f}min  {(el/(r_i+1)):.3f}s/row  eta {(n-r_i-1)*el/(r_i+1)/60:.1f}min",
                  flush=True)

    np.savez(os.path.join(outdir, f"{stem}.npz"), h_last=h_last, h_span=h_span, yesno=yesno,
             layers=np.array(A.layers))
    json.dump({"mode": A.mode, "split": A.split, "layers": A.layers, "n": n, "n_failed": bad,
               "model": A.model_path, "adapter": None, "max_pixels": HIGH_PX,
               "sys_prompt": SYS_GRADER if A.mode == "grader" else SYS_GEN,
               "minutes": round((time.time() - t0) / 60, 1), "rows": meta},
              open(os.path.join(outdir, f"{stem}.meta.json"), "w"))
    print(f"saved {outdir}/{stem}.npz  ({n} rows, {bad} failed, {(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
