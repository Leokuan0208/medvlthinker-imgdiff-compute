#!/usr/bin/env python3
"""lora_verifier_crossfit.py -- the July LoRA verifier, retrained on the probe's OWN pooled data (owed baseline).

The 2026-10-19 deck owes "LoRA verifier on pooled data": the July verifier (run_lora_verifier_open.py) was trained
on four domains, so comparing it with the pooled probe confounded method with data. This trains the July recipe
VERBATIM -- Lingshu-7B + LoRA r=16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down; AdamW lr 1e-4; bs 2 x accum 8;
1 epoch; the July system prompt and Yes/No question; loss on the final Yes/No token; full-resolution images (the
July default, cap_div 1) -- on fold f's fitting set from crossfit_data.fit_rows(f), which is final_method_eval.py's
fitting set row for row (115,934 / 116,000 rows; verified against final_eval_FINAL_cap320_Tmap.json).
A random 30,000 of those rows are used (seed 0): the full set would take ~10 GPU-hours per fold, and July found more
data barely moves this verifier (6k -> 10.4k: +0.005). Scoring: every distinct answer of every fold-f question in
the T=0.7 pool, batched per question, under HF (vLLM silently drops this adapter's 192 visual LoRA modules);
readout = argmax P(Yes), the July verifier's own. lora_verifier_compare.py scores the picks.

  CUDA_VISIBLE_DEVICES=0 python3 src/exp_2026_10/lora_verifier_crossfit.py --fold 0
"""
import argparse, json, math, os, random, sys, time
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossfit_data as CD                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True); ap.add_argument("--max_train", type=int, default=30000)
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--bs", type=int, default=2); ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lora_r", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smoke", type=int, default=0, help=">0: train on this many rows, score 25 questions of one cell, all into "
                "a scratch dir -- an end-to-end check that never touches the real adapter or records")
A = ap.parse_args()
MP = CD.PM.MP
OUTD = f"/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/train/lora_verifier_crossfit_1022/fold{A.fold}"
REC = f"/data/dan/exp_2026_10/lora_verifier_crossfit_fold{A.fold}.jsonl"
if A.smoke:
    OUTD = f"/data/dan/exp_2026_10/smoke/lora_verifier_fold{A.fold}"; REC = OUTD + ".jsonl"; A.max_train = A.smoke
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
MAXPX = HIGH_PX                                       # July default: cap_div 1
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
       "proposed answer is correct. Respond with only 'Yes' or 'No'.")   # run_lora_verifier_open.py, verbatim
random.seed(A.seed); np.random.seed(A.seed); torch.manual_seed(A.seed)
os.makedirs(OUTD, exist_ok=True); t0 = time.time()
log = lambda s: print(f"[{(time.time() - t0) / 60:6.1f} min] {s}", flush=True)

from transformers import AutoProcessor, AutoModelForImageTextToText   # noqa: E402
from qwen_vl_utils import process_vision_info                          # noqa: E402
from peft import LoraConfig, get_peft_model, PeftModel                 # noqa: E402

ev = CD.eval_images()
rows = CD.fit_rows(A.fold, ev)
log(f"fold {A.fold}: {len(rows):,} fitting rows (final_method_eval's set)")
rng = random.Random(A.seed); rng.shuffle(rows); rows = rows[:A.max_train]
train = []
for ds, r in rows:
    it = CD.items(ds).get(r["idx"])
    if it is not None:
        train.append((it[0], it[2], r["ans"], int(r["y"])))
log(f"{len(train):,} training examples (pos rate {np.mean([t[3] for t in train]):.3f})")

proc = AutoProcessor.from_pretrained(MP)
YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]; NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]


def msgs(q, img, proposed):
    body = f"Question: {q}\nProposed answer: {proposed}\nIs the proposed answer correct? Answer Yes or No."
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX},
                                         {"type": "text", "text": body}]}]


def encode(batch, labels=None):
    texts, ims = [], []
    for k, (q, img, ans) in enumerate(batch):
        m = msgs(q, img, ans); t = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        if labels is not None: t += "Yes" if labels[k] == 1 else "No"
        i, _ = process_vision_info(m); texts.append(t); ims += i
    return proc(text=texts, images=ims, return_tensors="pt", padding=True)


model = AutoModelForImageTextToText.from_pretrained(MP, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")
if os.path.exists(f"{OUTD}/adapter_config.json"):
    model = PeftModel.from_pretrained(model, OUTD); log("adapter found -- scoring only")
else:
    model = get_peft_model(model, LoraConfig(r=A.lora_r, lora_alpha=2 * A.lora_r, lora_dropout=0.05, bias="none",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)
    model.train(); step = 0
    for ep in range(A.epochs):
        rng.shuffle(train)
        for bi in range(0, len(train), A.bs):
            for q, img, ans, lab in train[bi:bi + A.bs]:      # one example per forward, as July
                try:
                    enc = encode([(q, img, ans)], [lab]).to("cuda")
                    lab_ids = enc["input_ids"].clone(); lab_ids[:, :-1] = -100
                    (model(**enc, labels=lab_ids).loss / (A.bs * A.accum)).backward()
                except Exception as e:
                    print(f"  skip: {str(e)[:80]}", flush=True)
            step += 1
            if step % A.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); opt.zero_grad()
            if step % 500 == 0:
                log(f"ep{ep} step {step}/{len(train) // A.bs}")
    model.save_pretrained(OUTD); log(f"adapter saved -> {OUTD}")
model.eval()
proc.tokenizer.padding_side = "left"                  # last position = each sequence's own last token
done = {(json.loads(l)["cell"], json.loads(l)["idx"]) for l in open(REC) if l.strip()} if os.path.exists(REC) else set()
CHECK = {"n": 0, "maxdiff": 0.0, "agree": 0}
with open(REC, "a") as fh:
    for c in CD.FME.OPEN8:
        byq = defaultdict(list)
        for r in CD.t07_rows(c):
            if CD.ER.half(r["img_md5"]) == A.fold:
                byq[r["idx"]].append(r)
        its = CD.items(c); n = 0
        for q, rs in byq.items():
            if (c, q) in done or q not in its:
                continue
            if A.smoke and n >= 25:
                break
            qq, _, img = its[q]
            with torch.no_grad():
                lg = model(**encode([(qq, img, r["ans"]) for r in rs]).to("cuda")).logits[:, -1].float()
            py = torch.softmax(lg[:, [YES, NO]], -1)[:, 0].cpu().numpy()
            k = int(np.argmax(py))
            if CHECK["n"] < 20 and len(rs) > 1:            # batched (left-padded) vs July's one-at-a-time scoring
                with torch.no_grad():
                    one = np.array([torch.softmax(model(**encode([(qq, img, r["ans"])]).to("cuda")).logits[0, -1].float()[[YES, NO]], -1)[0].item()
                                    for r in rs])
                CHECK["n"] += 1; CHECK["maxdiff"] = max(CHECK["maxdiff"], float(np.abs(one - py).max()))
                CHECK["agree"] += int(int(np.argmax(one)) == k)
                if CHECK["n"] == 20:
                    log(f"CHECK batched vs one-at-a-time: max |dP(Yes)| {CHECK['maxdiff']:.4f}, pick agreement {CHECK['agree']}/20")
                    assert CHECK["agree"] >= 19, "batched scoring changes the pick -- stop and score one at a time"
            fh.write(json.dumps({"cell": c, "idx": q, "fold": A.fold, "pick_ans": rs[k]["ans"], "pick_na": rs[k]["na"],
                                 "p_yes": [float(x) for x in py], "answers": [r["na"] for r in rs]}) + "\n"); n += 1
        fh.flush(); log(f"scored {c}: {n} questions")
        if A.smoke:
            break
log("DONE")
