#!/usr/bin/env python3
"""build_choicewhy_verifier_examples.py -- turn the N=8 MCQ candidate dumps into COMPOSITION-MATCHED
verifier training sets, one per format arm.

THE COMPARISON THIS SERVES.  The Phase-2 claim is about FORMAT: does a verifier trained on
"(choice)(why)" candidates (arm B2: "<letter>. <one-sentence finding>") select better than the same
verifier trained on bare-letter candidates (arm A)?  For format to be the only variable, the two
training sets must match on everything else:

  same questions      both arms are generated over the SAME disjoint pool manifest, item for item
  same size           --max_train examples in each arm
  same per-source mix identical per-source example counts in each arm (computed jointly below)
  same labeller       label = (extracted option letter == gold letter), the repo's MCQ grader, applied
                      identically to both arms (see the artifact's grader note)

WHY THE MIX IS COMPUTED JOINTLY.  A bare letter has at most `n_options` distinct surface forms, so arm A
yields ~1.4 unique candidates per question while arm B2 yields ~6.  A per-source quota that arm A cannot
fill would silently change the mix between arms.  So each source's quota is
    min(target_share * max_train, available_in_arm_A, available_in_arm_B2)
and any deficit is redistributed over the sources that still have headroom IN BOTH ARMS.  Every quota,
every shortfall and every redistribution is recorded.

TARGET SHARE defaults to the evaluation mix of the Phase-1 item set (SLAKE 416 / VQA-RAD 272 /
PMC-VQA 500 / MedXpert-MM 300 of 1,488).  MedXpertQA-MM has no public train split, so its 20.2% share is
assigned to pathvqa_closed_train and recorded as a shortfall.

DEDUPLICATION.  One example per unique NORMALIZED candidate string per question -- the same rule the
open-text verifier used (one example per unique normalized answer per question,
run_lora_verifier_disjoint.py).  The surface form kept is the first sample that produced it.

  python3 src/training_methods/build_choicewhy_verifier_examples.py
  -> data/choicewhy_mcq_split/verifier_examples_<arm>.jsonl
  -> results/cascade_methods/artifacts/choicewhy_verifier_examples.json
"""
import argparse, json, os, random, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cascade_methods"))
from choicewhy_common import ARM_NAME, extract, norm  # noqa: E402

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["A", "B2"])
ap.add_argument("--ckpt_dir", default="ckpts/choicewhy_train")
ap.add_argument("--manifest", default="data/choicewhy_mcq_split/train_items.jsonl")
ap.add_argument("--out_dir", default="data/choicewhy_mcq_split")
ap.add_argument("--out", default="results/cascade_methods/artifacts/choicewhy_verifier_examples.json")
ap.add_argument("--max_train", type=int, default=10364,
                help="total training examples per arm. Default = the size of ckpts/train/lora_verifier_disjoint.")
ap.add_argument("--n_samples", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pos_match_ref", default="A",
                help="reference arm for the POSITIVE-RATE-MATCHED variant; every other arm also gets a "
                     "'<arm>_posmatched' set with the reference's per-source positive/negative counts")
ap.add_argument("--emit_pos_matched", type=int, default=1)
A = ap.parse_args()

# evaluation mix of the Phase-1 item set -> target per-source shares
EVAL_MIX = {"SLAKE": 416, "VQA-RAD": 272, "PMC-VQA": 500, "MedXpert-MM": 300}
SRC_FOR_EVAL = {"SLAKE": "slake_closed_train", "VQA-RAD": "vqa_rad_closed_train",
                "PMC-VQA": "pmc_vqa_train", "MedXpert-MM": "pathvqa_closed_train"}
TOT = sum(EVAL_MIX.values())
TARGET_SHARE = {SRC_FOR_EVAL[k]: v / TOT for k, v in EVAL_MIX.items()}

MAN = [json.loads(l) for l in open(J(A.manifest)) if l.strip()]
BY_IDX = {(r["src"], r["idx"]): r for r in MAN}
SRCS = sorted({r["src"] for r in MAN})
print(f"[manifest] {len(MAN)} disjoint training questions across {SRCS}", flush=True)

# ------------------------------------------------------------------ candidates -> labelled examples
pool = {}          # arm -> src -> list of example dicts
qcov = {}          # arm -> src -> set of question ids that contributed
stats = {}
for arm in A.arms:
    an = ARM_NAME[arm]
    pool[arm], qcov[arm], stats[arm] = defaultdict(list), defaultdict(set), {}
    for src in SRCS:
        p = J(os.path.join(A.ckpt_dir, f"ckpt_{src}_{an}_sc{A.n_samples}.jsonl"))
        if not os.path.exists(p):
            print(f"  !! missing {p} -- source skipped for arm {arm}", flush=True)
            continue
        n_q = n_parsefail = 0
        uniq_per_q = []
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            man = BY_IDX[(r["src"], r["idx"])]
            seen = {}
            for s in r["raw_outputs"]:
                cand = s.strip()
                if not cand:
                    continue
                key = norm(cand)
                if key in seen:
                    continue
                letter, ok, _rule = extract(cand, an)
                if not ok:
                    n_parsefail += 1
                    continue
                seen[key] = None
                pool[arm][src].append({
                    "arm": an, "src": src, "family": r["family"], "idx": r["idx"],
                    "question": r["question"], "options": r["options"], "gold": r["gold"],
                    "img_path": man["img_path"], "image_md5_rgb": r["image_md5_rgb"],
                    "candidate": cand, "letter": letter, "label": int(letter == r["gold"]),
                })
                qcov[arm][src].add(r["idx"])
            n_q += 1
            uniq_per_q.append(len(seen))
        stats[arm][src] = {
            "questions": n_q, "examples_available": len(pool[arm][src]),
            "unique_candidates_per_question": round(sum(uniq_per_q) / max(1, len(uniq_per_q)), 3),
            "unparsable_samples_dropped": n_parsefail,
            "pos_rate_available": round(sum(e["label"] for e in pool[arm][src]) / max(1, len(pool[arm][src])), 4),
        }
        s = stats[arm][src]
        print(f"  arm {arm:2s} {src:22s} q={s['questions']:5d} uniq/q={s['unique_candidates_per_question']:.2f} "
              f"-> {s['examples_available']:6d} examples (pos {s['pos_rate_available']:.3f}, "
              f"{s['unparsable_samples_dropped']} unparsable dropped)", flush=True)

# ------------------------------------------------------------------ joint quota
avail = {src: min(len(pool[arm].get(src, [])) for arm in A.arms) for src in SRCS}
quota = {src: min(int(round(TARGET_SHARE.get(src, 0.0) * A.max_train)), avail[src]) for src in SRCS}
short = {src: int(round(TARGET_SHARE.get(src, 0.0) * A.max_train)) - quota[src]
         for src in SRCS if int(round(TARGET_SHARE.get(src, 0.0) * A.max_train)) > quota[src]}
# redistribute the deficit over sources with headroom in BOTH arms, proportional to target share
deficit = A.max_train - sum(quota.values())
redistributed = {}
while deficit > 0:
    head = [s for s in SRCS if avail[s] > quota[s]]
    if not head:
        break
    w = {s: TARGET_SHARE.get(s, 1e-9) for s in head}
    wsum = sum(w.values())
    moved = 0
    for s in head:
        add = min(avail[s] - quota[s], max(1, int(deficit * w[s] / wsum)))
        add = min(add, deficit - moved)
        if add <= 0:
            continue
        quota[s] += add
        redistributed[s] = redistributed.get(s, 0) + add
        moved += add
        if moved >= deficit:
            break
    if moved == 0:
        break
    deficit -= moved
final_total = sum(quota.values())
print(f"\n[quota] target share {({k: round(v,4) for k,v in TARGET_SHARE.items()})}", flush=True)
print(f"[quota] available (min over arms) {avail}", flush=True)
print(f"[quota] FINAL per-source example quota {quota}  total={final_total} (target {A.max_train})", flush=True)
print(f"[quota] shortfalls vs eval-proportional target {short}; redistributed {redistributed}", flush=True)

# ------------------------------------------------------------------ draw, identically for every arm
report = {}
os.makedirs(J(A.out_dir), exist_ok=True)
for arm in A.arms:
    rng = random.Random(A.seed)
    ex = []
    for src in SRCS:
        p = list(pool[arm].get(src, []))
        p.sort(key=lambda e: (str(e["idx"]), e["candidate"]))   # deterministic before shuffling
        rng.shuffle(p)
        ex += p[:quota[src]]
    rng.shuffle(ex)
    op = J(os.path.join(A.out_dir, f"verifier_examples_{ARM_NAME[arm]}.jsonl"))
    with open(op, "w") as fh:
        for e in ex:
            fh.write(json.dumps(e) + "\n")
    pos = sum(e["label"] for e in ex) / max(1, len(ex))
    per_src_pos = {s: round(sum(e["label"] for e in ex if e["src"] == s) /
                            max(1, sum(1 for e in ex if e["src"] == s)), 4) for s in SRCS}
    report[ARM_NAME[arm]] = {
        "file": os.path.relpath(op, ROOT),
        "n_examples": len(ex),
        "per_source_examples": dict(Counter(e["src"] for e in ex)),
        "distinct_questions": len({(e["src"], e["idx"]) for e in ex}),
        "distinct_questions_per_source": {s: len({e["idx"] for e in ex if e["src"] == s}) for s in SRCS},
        "distinct_images": len({e["image_md5_rgb"] for e in ex}),
        "pos_label_rate": round(pos, 4),
        "pos_label_rate_per_source": per_src_pos,
        "mean_candidate_chars": round(sum(len(e["candidate"]) for e in ex) / max(1, len(ex)), 1),
        "mean_candidate_words": round(sum(len(e["candidate"].split()) for e in ex) / max(1, len(ex)), 2),
        "availability": stats[arm],
    }
    r = report[ARM_NAME[arm]]
    print(f"\n[arm {arm}] {r['n_examples']} examples | {r['distinct_questions']} questions | "
          f"{r['distinct_images']} images | POS RATE {r['pos_label_rate']} | "
          f"mean candidate {r['mean_candidate_words']} words -> {r['file']}", flush=True)
    print(f"         per-source {r['per_source_examples']}", flush=True)
    print(f"         per-source pos rate {r['pos_label_rate_per_source']}", flush=True)

# ------------------------------------------------------------------ positive-rate-matched variant
# WHY.  Deduplicating by unique candidate STRING gives arm A at most n_options examples per question,
# while arm B2 gets ~6 distinct phrasings -- and the correct option tends to attract more distinct
# phrasings, so B2's label base rate comes out higher than A's for the SAME questions.  A base-rate
# shift moves a verifier's operating point independently of its discrimination.  This variant removes
# that confound by drawing, per source, exactly the reference arm's positive and negative COUNTS.
posmatch = {}
if A.emit_pos_matched and A.pos_match_ref in A.arms:
    ref = ARM_NAME[A.pos_match_ref]
    ref_counts = {}
    ref_ex = [json.loads(l) for l in open(J(os.path.join(A.out_dir, f"verifier_examples_{ref}.jsonl")))
              if l.strip()]
    for s in SRCS:
        rr = [e for e in ref_ex if e["src"] == s]
        ref_counts[s] = {1: sum(e["label"] for e in rr), 0: sum(1 - e["label"] for e in rr)}
    print(f"\n[pos-match] reference arm {ref} per-source (pos,neg) counts: "
          f"{ {s: (ref_counts[s][1], ref_counts[s][0]) for s in SRCS} }", flush=True)
    for arm in A.arms:
        if arm == A.pos_match_ref:
            continue
        rng = random.Random(A.seed)
        ex, feasible = [], {}
        for s in SRCS:
            for lab in (1, 0):
                cand = [e for e in pool[arm].get(s, []) if e["label"] == lab]
                cand.sort(key=lambda e: (str(e["idx"]), e["candidate"]))
                rng.shuffle(cand)
                want = ref_counts[s][lab]
                ex += cand[:want]
                feasible[f"{s}|{'pos' if lab else 'neg'}"] = {"wanted": want, "available": len(cand),
                                                              "taken": min(want, len(cand))}
        rng.shuffle(ex)
        op = J(os.path.join(A.out_dir, f"verifier_examples_{ARM_NAME[arm]}_posmatched.jsonl"))
        with open(op, "w") as fh:
            for e in ex:
                fh.write(json.dumps(e) + "\n")
        pm = {"file": os.path.relpath(op, ROOT), "n_examples": len(ex),
              "per_source_examples": dict(Counter(e["src"] for e in ex)),
              "distinct_questions": len({(e["src"], e["idx"]) for e in ex}),
              "distinct_images": len({e["image_md5_rgb"] for e in ex}),
              "pos_label_rate": round(sum(e["label"] for e in ex) / max(1, len(ex)), 4),
              "pos_label_rate_per_source": {s: round(sum(e["label"] for e in ex if e["src"] == s) /
                                                     max(1, sum(1 for e in ex if e["src"] == s)), 4)
                                            for s in SRCS},
              "mean_candidate_words": round(sum(len(e["candidate"].split()) for e in ex) / max(1, len(ex)), 2),
              "draw_feasibility": feasible,
              "reference_arm": ref}
        posmatch[ARM_NAME[arm] + "_posmatched"] = pm
        report[ARM_NAME[arm] + "_posmatched"] = pm
        print(f"[pos-match] arm {arm}: {pm['n_examples']} examples | POS RATE {pm['pos_label_rate']} "
              f"(reference {report[ref]['pos_label_rate']}) | {pm['distinct_questions']} questions -> "
              f"{pm['file']}", flush=True)

comp = {a: report[a]["per_source_examples"] for a in report}
matched = len({json.dumps(v, sort_keys=True) for v in comp.values()}) == 1
sizes = {a: report[a]["n_examples"] for a in report}
print(f"\n[composition-match] per-source example counts identical across arms: {matched}; sizes {sizes}")

out = {
    "purpose": "composition-matched verifier training sets for the MCQ (choice)(why) format comparison",
    "date": "2026-08-03",
    "builder": "src/training_methods/build_choicewhy_verifier_examples.py",
    "candidates": f"{A.ckpt_dir}/ckpt_<src>_<arm>_sc{A.n_samples}.jsonl (Lingshu-7B, n=8, temp 0.7, seed 1234)",
    "manifest": A.manifest,
    "label_rule": "label = int(extracted option letter == gold option letter). This IS the repo's MCQ "
                  "grader: MCQ scoring in this project is exact letter match (src/labeling/run_vlm_eval.py, "
                  "the Phase-1 pilot, ckpts/gate_lingshu7b_mcq), and the same rule labels the training "
                  "candidates and grades the evaluation candidates, so training and evaluation share one "
                  "grader. The free-text LLM judge (src/labeling/run_judge.py) grades open-ended answers "
                  "and is not the grader for any MCQ number in this repo; a concordance audit against it "
                  "is reported separately.",
    "extractor": "src/cascade_methods/choicewhy_common.py::extract (verbatim from the Phase-1 analyzer)",
    "dedup": "one example per unique normalized candidate string per question",
    "target_share_from_eval_mix": {k: round(v, 4) for k, v in TARGET_SHARE.items()},
    "eval_mix_items": EVAL_MIX,
    "medxpert_surrogate": "MedXpertQA-MM has no public train split; its share is assigned to "
                          "pathvqa_closed_train and recorded here as a substitution, not a match.",
    "available_examples_min_over_arms": avail,
    "final_per_source_quota": quota,
    "quota_shortfalls_vs_eval_proportional": short,
    "quota_redistributed": redistributed,
    "max_train_target": A.max_train,
    "final_total_per_arm": final_total,
    "composition_matched_across_arms": bool(matched),
    "pos_rate_matched_variant": {
        "why": "dedup by unique candidate STRING caps arm A at n_options examples per question while "
               "arm B2 gets ~6 phrasings, and correct options attract more distinct phrasings -- so the "
               "label base rate is higher in B2 for the SAME questions. A base-rate shift moves a "
               "verifier's operating point independently of its discrimination, so a variant with the "
               "reference arm's exact per-source positive/negative counts is emitted alongside.",
        "reference_arm": ARM_NAME.get(A.pos_match_ref),
        "sets": posmatch,
    },
    "arms": report,
    "seed": A.seed,
}
os.makedirs(os.path.dirname(J(A.out)), exist_ok=True)
json.dump(out, open(J(A.out), "w"), indent=1)
print(f"wrote -> {A.out}")
