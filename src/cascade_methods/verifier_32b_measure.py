#!/usr/bin/env python3
"""
verifier_32b_measure.py -- OFFLINE measure for the "does a stronger verifier break the wall?" test.

Reads the raw per-question verdicts written by verifier_32b_gpu.py for each model/dataset and computes
SELECTION accuracy over DISTINCT candidate answers for each verifier, versus oracle@distinct and the
SC-majority / greedy references. All arms use the SAME questions, the SAME distinct candidate sets, and
the SAME exact-match `ok` labels -> apples-to-apples; the only thing that varies is the verifier that
scores the candidates.

Verifiers compared (whichever ckpts are present):
  * 7B-trained  : `s7b` per group (Lingshu-7B + pooled4 LoRA pointwise verifier, from the diverse dump).
  * 7B-zeroshot : `p_yes` from ckpt tag lingshu7b_zs (base Lingshu-7B, same prompt).  [capacity control]
  * 32B-zeroshot: `p_yes` from ckpt tag lingshu32b  (base Lingshu-32B, same prompt).  [the capacity attack]

Metrics per dataset + pooled: greedy, SC-majority, oracle@distinct, each verifier's selection accuracy,
selection EFFICIENCY = P(pick correct | a correct candidate exists), and conversion =
(sel - SC)/(oracle - SC). Paired bootstrap 95% CI on (32B - 7B_trained) and (32B - 7B_zeroshot).

No GPU. No fabricated numbers. Launch from repo root:
  python3 src/cascade_methods/verifier_32b_measure.py
Writes: results/cascade_methods/artifacts/verifier_32b_gpu.json
"""
import os, json, glob
import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
CK = "ckpts/openvqa/verifier32b"
DATASETS = ["vqa_rad_open", "slake_open", "pmc_content"]
# ckpt tag -> verifier label + which score field to argmax
VERIFIERS = {
    "7B-trained":   ("lingshu32b", "s7b"),     # s7b is carried in every ckpt; read from the 32B ckpt
    "7B-zeroshot":  ("lingshu7b_zs", "p_yes"),
    "32B-zeroshot": ("lingshu32b", "p_yes"),
}


def load_ckpt(ds, tag):
    p = J(os.path.join(CK, f"ckpt_{ds}_{tag}.jsonl"))
    if not os.path.exists(p): return None
    return {str(json.loads(l)["idx"]): json.loads(l) for l in open(p) if l.strip()}


def sel_ok(groups, field):
    """selection accuracy contribution for one question: ok of the argmax-`field` group."""
    j = int(np.argmax([g[field] for g in groups]))
    return int(groups[j]["ok"])


def boot_ci(a, b, n_boot=5000, seed=0):
    """paired bootstrap 95% CI of mean(a)-mean(b) over questions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed); n = len(a); d = a - b
    means = [d[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def measure(per_q):
    """per_q: dict tag->list aligned per question. Returns metric dict."""
    n = len(per_q["oracle"])
    out = {"n": n,
           "greedy": float(np.mean(per_q["greedy"])),
           "sc_majority": float(np.mean(per_q["modal"])),
           "oracle_distinct": float(np.mean(per_q["oracle"])),
           "mean_distinct": float(np.mean(per_q["ndist"])),
           "verifiers": {}}
    orc = np.asarray(per_q["oracle"])
    for lab in VERIFIERS:
        if lab not in per_q: continue
        sel = np.asarray(per_q[lab], float)
        sel_acc = float(sel.mean())
        # selection efficiency: P(pick correct | correct candidate exists)
        mask = orc == 1
        eff = float(sel[mask].mean()) if mask.any() else float("nan")
        denom = out["oracle_distinct"] - out["sc_majority"]
        conv = float((sel_acc - out["sc_majority"]) / denom) if abs(denom) > 1e-9 else float("nan")
        out["verifiers"][lab] = {"sel_acc": sel_acc, "sel_efficiency": eff,
                                 "conversion_of_SC_to_oracle_headroom": conv}
    # paired deltas vs 32B
    if "32B-zeroshot" in per_q:
        out["deltas_vs_32B"] = {}
        for lab in ("7B-trained", "7B-zeroshot"):
            if lab in per_q:
                d, lo, hi = boot_ci(per_q["32B-zeroshot"], per_q[lab])
                out["deltas_vs_32B"][f"32B_minus_{lab}"] = {"mean": d, "ci95": [lo, hi],
                                                            "significant": bool(lo > 0 or hi < 0)}
    return out


def build_perq(cks):
    """Align verdicts across models by idx (intersection) and build per-question arrays."""
    base_tag = "lingshu32b"
    if cks.get(base_tag) is None: return None
    common = set(cks[base_tag])
    for tag in set(t for _, (t, _) in VERIFIERS.items()):
        if cks.get(tag) is not None:
            common &= set(cks[tag])
    ids = sorted(common, key=lambda x: (len(x), x))
    pq = {"oracle": [], "modal": [], "greedy": [], "ndist": []}
    for lab, (tag, field) in VERIFIERS.items():
        if cks.get(tag) is None: continue
        pq[lab] = []
    for i in ids:
        r32 = cks[base_tag][i]
        pq["oracle"].append(r32["oracle"]); pq["modal"].append(r32["modal_ok"])
        pq["greedy"].append(r32["greedy_ok"]); pq["ndist"].append(r32["n_distinct"])
        for lab, (tag, field) in VERIFIERS.items():
            if cks.get(tag) is None: continue
            groups = cks[tag][i]["groups"]
            # sanity: same distinct count across models (deterministic grouping)
            pq[lab].append(sel_ok(groups, field))
    return pq, ids


def main():
    result = {"experiment": "stronger (zero-shot) 32B verifier vs trained-7B verifier: does verifier "
                            "CAPACITY break the selectability wall?",
              "labels": "exact-match/substring oks (map_correct) -- conservative, applied identically to "
                        "every verifier and to oracle. NOT LLM-judge; relative deltas are the valid read.",
              "candidate_pool": "Lingshu-7B diverse pool (5-prompt x 3-temp, M<=15) from "
                                "ckpts/openvqa/diverse; deduped to DISTINCT answers per question.",
              "confound": "7B-trained = pooled4 LoRA (trained); 7B-zeroshot & 32B-zeroshot = base models, "
                          "same VERIFY_SYS prompt. Pure capacity read = 32B-zeroshot vs 7B-zeroshot.",
              "per_dataset": {}, "pooled": {}, "missing": []}
    pooled = None
    for ds in DATASETS:
        cks = {tag: load_ckpt(ds, tag) for tag in set(t for _, (t, _) in VERIFIERS.items())}
        if cks.get("lingshu32b") is None:
            result["per_dataset"][ds] = {"status": "32B verdicts MISSING (tp=2-blocked or not run)"}
            result["missing"].append(f"{ds}:lingshu32b")
            continue
        pq, ids = build_perq(cks)
        result["per_dataset"][ds] = measure(pq)
        result["per_dataset"][ds]["verifiers_present"] = [l for l in VERIFIERS if l in pq]
        if pooled is None:
            pooled = {k: list(v) for k, v in pq.items()}
        else:
            for k in pq:
                pooled.setdefault(k, []); pooled[k].extend(pq[k])
    if pooled is not None:
        result["pooled"] = measure(pooled)
        result["pooled"]["datasets"] = [d for d in DATASETS if "n" in result["per_dataset"].get(d, {})]
        # verdict
        v = result["pooled"]["verifiers"]; d = result["pooled"].get("deltas_vs_32B", {})
        verdict = []
        if "32B-zeroshot" in v and "7B-trained" in v:
            dd = d.get("32B_minus_7B-trained", {})
            verdict.append(f"32B-zeroshot sel={v['32B-zeroshot']['sel_acc']:.3f} vs "
                           f"7B-trained sel={v['7B-trained']['sel_acc']:.3f} "
                           f"(delta={dd.get('mean',float('nan')):+.3f}, CI{dd.get('ci95')}, "
                           f"sig={dd.get('significant')}).")
        if "32B-zeroshot" in v and "7B-zeroshot" in v:
            dd = d.get("32B_minus_7B-zeroshot", {})
            verdict.append(f"Pure capacity: 32B-zeroshot vs 7B-zeroshot "
                           f"delta={dd.get('mean',float('nan')):+.3f} CI{dd.get('ci95')} sig={dd.get('significant')}.")
        if "32B-zeroshot" in v:
            gap = result["pooled"]["oracle_distinct"] - v["32B-zeroshot"]["sel_acc"]
            verdict.append(f"32B leaves an oracle->selection gap of {gap:.3f} "
                           f"(oracle={result['pooled']['oracle_distinct']:.3f}); "
                           f"conversion={v['32B-zeroshot']['conversion_of_SC_to_oracle_headroom']:.2f}.")
        result["verdict_summary"] = " ".join(verdict)
    outp = J("results/cascade_methods/artifacts/verifier_32b_gpu.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(result, open(outp, "w"), indent=2)
    print(json.dumps(result, indent=2))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
