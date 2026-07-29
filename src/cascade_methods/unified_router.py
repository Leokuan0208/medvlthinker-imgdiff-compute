#!/usr/bin/env python3
"""
unified_router.py - Method C: a deterministic FORMAT-AWARE ROUTER over a pooled MCQ+open-text workload.

There is no runtime learned router in this project: MCQ and open-text are run as two separate,
condition-aware pipelines.  This script builds the missing piece -- a single deterministic router that,
given a MIXED stream of questions, detects each item's answer format FROM THE PROMPT TEXT (never from the
gold answer) and dispatches it to the method built for that format, then scores the WHOLE pooled set as
ONE accuracy-vs-cost number against the always-strong baseline, per family.

Routing rules (detect_format):
  1. enumerated options present  ('Options:' / 'Answer Choices:' / >=3 'A. / (A) / A:' markers /
     a non-empty `choices` list)                    -> MCQ   -> logprob-MARGIN cascade
  2. else, a yes/no interrogative  (is/are/does/...) -> CLOSED -> logprob-MARGIN cascade (2-way margin)
  3. else                                            -> OPEN   -> trained best-of-8 VERIFIER cascade

MCQ/CLOSED path  (src/cascade_methods/{acc.py,cascade_all_families.py}): cheap 7B answers; escalate to the
  strong 32B/38B when the cheap answer-letter margin (top1-top2 logprob) < tau_mcq.  Per-sample data:
  MedEvalKit/eval_results_<model>/{}/<BENCH>/results.json  (fields: prompt/question, choices, margin,
  correct, gen_toks).
OPEN path        (src/cascade_methods/open_verifier_cascade_table.py): cheap 7B samples N=8 free-text
  answers, the trained LoRA verifier picks argmax P(Yes); escalate to strong when verifier-confidence
  (max P(Yes)) < tau_open.  Per-sample data: ckpts/train/lora_verifier_pooled4/transfer_dump_<ds>_<tag>.json
  (sl[8], scores[8], pick, greedy_ok) + strong judge ckpts/openvqa/<strong>/ckpt_<ds>_<stag>*.judge.jsonl.

Because MedEvalKit-'open' items (SLAKE-open, PATH_VQA free-text) have no best-of-8 verifier outputs at those
exact indices, the OPEN contribution to the pool is measured on the dedicated open-eval sets
(kvasir/vqa_rad_open/pathvqa_open/radimagenet_open), which are the same benchmarks in open form.  MedEvalKit
items the detector labels 'open' are therefore EXCLUDED from the MCQ pool (logged) rather than double-counted.

Operating point: honest 50/50 stratified calib/test, 20 seeds.  On calib we jointly pick (tau_mcq, tau_open)
= the min-pooled-cost thresholds s.t. pooled calib accuracy >= pooled always-strong accuracy; evaluate on
test.  Cost = prefill-inclusive FLOPs 2*N*(P+G); best-of-8 counted at full 8x cheap passes (repo convention,
conservative -- prefix caching shares the image prefill, so the PREFIX-SHARED row is the deployable lower bound).
CPU only; launch from repo root.
"""
import os, sys, json, glob, re
import numpy as np
from collections import defaultdict, Counter

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)

N7, N32 = 7.6e9, 33.0e9          # cheap / strong backbone params (harness.py)
PFIX = 420.0                     # fixed prompt-token estimate; prefill-bound (P>>G) so the cheap/strong ratio is robust
BEST_OF = int(os.environ.get("N_OPEN", "8"))   # samples drawn for the open verifier leg (lever: bo2/bo3 already beat strong)

MCQ_BENCHES = ["PMC_VQA", "MedXpertQA-MM", "SLAKE", "VQA_RAD", "PATH_VQA"]
OPEN_DSETS  = ["kvasir_open", "vqa_rad_open", "pathvqa_open", "radimagenet_open"]

# family -> (cheap MCQ dir, [strong MCQ dir candidates], open verifier tag, strong-open dir, [strong-open judge name templates])
FAMILIES = {
    "Lingshu": dict(
        mcq_cheap="eval_results_lingshu7b_full",
        mcq_strong=["eval_results_lingshu32b_full", "eval_results_lingshu32b_reason"],
        vtag="lingshu7b", open_strong="ckpts/openvqa/strong_lingshu",
        open_sjudge=["ckpt_{ds}_lingshu32b_t0.judge.jsonl", "ckpt_{ds}_lingshu32b.judge.jsonl"]),
    "MedVLThinker": dict(
        mcq_cheap="eval_results_mvt7b",
        mcq_strong=["eval_results_mvt32b", "eval_results_mvt32b_reason"],
        vtag="7b", open_strong="ckpts/openvqa/strong",
        open_sjudge=["ckpt_{ds}_32b_t0.judge.jsonl", "ckpt_{ds}_32b.judge.jsonl"]),
    "InternVL3": dict(
        mcq_cheap="eval_results_iv3_8b",
        mcq_strong=["eval_results_iv3_38b", "eval_results_iv3_38b_reason"],
        vtag="iv3_8b", open_strong="ckpts/openvqa/internvl3_38b",
        open_sjudge=["ckpt_{ds}_iv3_38b_t0.judge.jsonl", "ckpt_{ds}_iv3_38b.judge.jsonl"]),
}
ADAPTER = "ckpts/train/lora_verifier_pooled4"

# ----------------------------- format detection (from PROMPT text only) -----------------------------
_YN = re.compile(r'^(is|are|does|do|did|was|were|has|have|had|can|could|should|shall|will|would|may|might)\b')
_OPT = re.compile(r'(?m)(?:^|[\s\(])([A-Ea-e])(?:\)|[:.])\s')

def detect_format(item):
    text = (item.get("prompt") or item.get("question") or "")
    ch = item.get("choices")
    has_choices = isinstance(ch, list) and len(ch) >= 2
    low = text.lower()
    if ("options:" in low) or ("answer choices" in low) or has_choices or len(set(_OPT.findall(text))) >= 3:
        return "mcq"
    q = re.sub(r"\n.*$", "", text.strip().lower()).strip()   # first line
    if _YN.match(q):
        return "closed"
    return "open"

# ----------------------------- loaders -----------------------------
def _mcq_rows(md, ds):
    fs = glob.glob(J(f"MedEvalKit/{md}/*/{ds}/results.json"))
    return json.load(open(fs[0])) if fs else None

def _okv(r):
    return int(bool(r.get("correct")))

def _loadj(p):
    m = {}
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); m[r["idx"]] = r
    return m

def _load_judge(p):
    m = {}
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); m[r["idx"]] = int(r["judge_ok"])
    return m

def load_mcq(cfg):
    """Return per-item arrays for detector-MCQ/CLOSED items only (open MedEvalKit items excluded, counted)."""
    out = defaultdict(list)  # keys: ok_c, ok_s, margin, g_c, g_s, bench, fmt
    catch = Counter(); excluded = Counter()
    for ds in MCQ_BENCHES:
        rc = _mcq_rows(cfg["mcq_cheap"], ds)
        rs = None
        for sd in cfg["mcq_strong"]:
            rs = _mcq_rows(sd, ds)
            if rs is not None: break
        if rc is None or rs is None:
            continue
        n = min(len(rc), len(rs))
        for i in range(n):
            f = detect_format(rc[i])
            catch[f] += 1
            if f == "open":
                excluded[ds] += 1
                continue
            out["ok_c"].append(_okv(rc[i])); out["ok_s"].append(_okv(rs[i]))
            out["margin"].append(float(rc[i].get("margin") or 0.0))
            out["g_c"].append(float(rc[i].get("gen_toks") or 3))
            out["g_s"].append(float(rs[i].get("gen_toks") or 3))
            out["l_c"].append(float(rc[i].get("latency_s") or 0.0))
            out["l_s"].append(float(rs[i].get("latency_s") or 0.0))
            out["bench"].append(ds); out["fmt"].append(f)
    A = {k: (np.array(v) if k not in ("bench", "fmt") else np.array(v, dtype=object)) for k, v in out.items()}
    return A, catch, excluded

def load_open(cfg):
    out = defaultdict(list)  # ok_v(bo8 pick), ok_s(strong judge), gate(max score), g_c(sum-8 gen), g_s, bench
    catch = Counter()
    for ds in OPEN_DSETS:
        dp = J(os.path.join(ADAPTER, f"transfer_dump_{ds}_{cfg['vtag']}.json"))
        if not os.path.exists(dp):
            continue
        dump = json.load(open(dp))
        sj = {}
        for tmpl in cfg["open_sjudge"]:
            sj = _load_judge(J(os.path.join(cfg["open_strong"], tmpl.format(ds=ds))))
            if sj: break
        # cheap sc8 (for gen-token accounting; optional)
        for r in dump:
            i = r["idx"]
            if i not in sj:
                continue
            sl = [0 if x is None or x == -1 else int(x) for x in r["sl"]]
            k = min(BEST_OF, len(r["scores"]))               # best-of-BEST_OF over the first k samples
            pick = int(np.argmax(r["scores"][:k]))
            out["ok_v"].append(int(sl[pick]))
            out["ok_s"].append(int(sj[i]))
            out["gate"].append(float(max(r["scores"][:k])))
            out["g_c"].append(6.0)      # short free-text answers; prefill-bound (see PFIX note)
            out["g_s"].append(6.0)
            out["bench"].append(ds)
            catch["open"] += 1
    A = {k: (np.array(v) if k != "bench" else np.array(v, dtype=object)) for k, v in out.items()}
    return A, catch

# ----------------------------- cost model (prefill-inclusive FLOPs = 2N(P+G)) -----------------------------
def mcq_cost_cheap(g):  return 2 * N7 * (PFIX + g)
def mcq_cost_strong(g): return 2 * N32 * (PFIX + g)
def open_cost_cheap(g, shared=False):
    # best-of-8 sampling + verifier scoring on the cheap 7B.  Conservative: 8x full passes (repo convention,
    # verifier FLOPs folded in as negligible LoRA cost).  shared=True: prefix-cached -> 1 prefill + 8 decodes.
    if shared:
        return 2 * N7 * (PFIX + BEST_OF * g)
    return BEST_OF * 2 * N7 * (PFIX + g)
def open_cost_strong(g): return 2 * N32 * (PFIX + g)

# ----------------------------- evaluation -----------------------------
def strat_key(bench, ok_c, ok_s):
    return np.array([f"{b}|{int(a)}{int(c)}" for b, a, c in zip(bench, ok_c, ok_s)])

def eval_family(name, cfg, seeds=range(20), exclude_medxpert=False):
    M, catch_m, excl = load_mcq(cfg)
    O, catch_o = load_open(cfg)
    if len(M.get("ok_c", [])) == 0 and len(O.get("ok_v", [])) == 0:
        print(f"\n### {name}: NO DATA"); return None

    # optional MedXpert drop (near-chance for both models; excluded from project's main claims)
    if exclude_medxpert and len(M["ok_c"]):
        keep = M["bench"] != "MedXpertQA-MM"
        M = {k: v[keep] for k, v in M.items()}

    nM = len(M.get("ok_c", [])); nO = len(O.get("ok_v", []))
    # per-item FLOPs cost vectors
    cM_cheap = np.array([mcq_cost_cheap(g) for g in M["g_c"]]) if nM else np.array([])
    cM_str   = np.array([mcq_cost_strong(g) for g in M["g_s"]]) if nM else np.array([])
    cO_cheap = np.array([open_cost_cheap(g) for g in O["g_c"]]) if nO else np.array([])
    cO_cheapS= np.array([open_cost_cheap(g, shared=True) for g in O["g_c"]]) if nO else np.array([])
    cO_str   = np.array([open_cost_strong(g) for g in O["g_s"]]) if nO else np.array([])
    # per-item LATENCY vectors (amortized-batch proxy from latency_s; batch-1 measurement is queued as a GPU job).
    # MCQ: per-sample latency_s. OPEN: best-of-N samples run in PARALLEL -> cheap leg ~ ONE cheap-pass latency
    # (backbone proxy = mean MCQ cheap latency); strong open ~ mean MCQ strong latency (same backbones).
    lM_cheap = np.array(M["l_c"]) if nM else np.array([])
    lM_str   = np.array(M["l_s"]) if nM else np.array([])
    L7  = float(np.mean(lM_cheap)) if nM and lM_cheap.mean() > 0 else 0.15
    L32 = float(np.mean(lM_str))   if nM and lM_str.mean()  > 0 else 0.30
    lO_cheap = np.full(nO, L7); lO_str = np.full(nO, L32)

    # always-strong pooled baseline (accuracy + cost) over the SAME pooled test items -> computed per split
    okM_c, okM_s, mar = (M["ok_c"], M["ok_s"], M["margin"]) if nM else (np.array([]),)*3
    okO_v, okO_s, gat = (O["ok_v"], O["ok_s"], O["gate"]) if nO else (np.array([]),)*3

    agg = defaultdict(list)
    for s in seeds:
        rng = np.random.default_rng(s)
        # stratified 50/50 for each pool
        calM = np.zeros(nM, bool); calO = np.zeros(nO, bool)
        if nM:
            for k in np.unique(strat_key(M["bench"], okM_c, okM_s)):
                ix = np.where(strat_key(M["bench"], okM_c, okM_s) == k)[0]; rng.shuffle(ix); calM[ix[:len(ix)//2]] = True
        if nO:
            for k in np.unique(strat_key(O["bench"], okO_v, okO_s)):
                ix = np.where(strat_key(O["bench"], okO_v, okO_s) == k)[0]; rng.shuffle(ix); calO[ix[:len(ix)//2]] = True
        teM, teO = ~calM, ~calO

        # pooled always-strong parity on calib
        nMc = calM.sum(); nOc = calO.sum()
        strong_correct_cal = (okM_s[calM].sum() if nM else 0) + (okO_s[calO].sum() if nO else 0)
        parity_cal = strong_correct_cal / max(nMc + nOc, 1)

        # candidate thresholds
        qM = np.quantile(mar[calM], np.linspace(0, 1, 26)) if nM else [0.0]
        qO = np.quantile(gat[calO], np.linspace(0, 1, 26)) if nO else [0.0]

        def pooled_calib(tm, to):
            corr = 0.0; cost = 0.0
            if nM:
                esc = mar[calM] < tm
                fin = np.where(esc, okM_s[calM], okM_c[calM])
                corr += fin.sum(); cost += (cM_cheap[calM] + np.where(esc, cM_str[calM], 0)).sum()
            if nO:
                esc = gat[calO] < to
                fin = np.where(esc, okO_s[calO], okO_v[calO])
                corr += fin.sum(); cost += (cO_cheap[calO] + np.where(esc, cO_str[calO], 0)).sum()
            return corr / max(nMc + nOc, 1), cost

        best = None
        for tm in qM:
            for to in qO:
                acc, cost = pooled_calib(tm, to)
                if acc >= parity_cal - 1e-9 and (best is None or cost < best[0]):
                    best = (cost, tm, to)
        tm, to = (best[1], best[2]) if best else (qM[0], qO[0])

        # evaluate on TEST
        nMt = teM.sum(); nOt = teO.sum()
        corr = 0.0; cost = 0.0; costS = 0.0; base_cost = 0.0; base_corr = 0.0
        lat = 0.0; base_lat = 0.0
        escM = escO = 0.0
        cheap_only_corr = 0.0
        if nM:
            esc = mar[teM] < tm
            fin = np.where(esc, okM_s[teM], okM_c[teM])
            corr += fin.sum(); base_corr += okM_s[teM].sum(); cheap_only_corr += okM_c[teM].sum()
            cost += (cM_cheap[teM] + np.where(esc, cM_str[teM], 0)).sum()
            costS += (cM_cheap[teM] + np.where(esc, cM_str[teM], 0)).sum()
            lat += (lM_cheap[teM] + np.where(esc, lM_str[teM], 0)).sum()
            base_cost += cM_str[teM].sum(); base_lat += lM_str[teM].sum(); escM = esc.mean()
        if nO:
            esc = gat[teO] < to
            fin = np.where(esc, okO_s[teO], okO_v[teO])
            corr += fin.sum(); base_corr += okO_s[teO].sum(); cheap_only_corr += okO_v[teO].sum()
            cost += (cO_cheap[teO] + np.where(esc, cO_str[teO], 0)).sum()
            costS += (cO_cheapS[teO] + np.where(esc, cO_str[teO], 0)).sum()
            lat += (lO_cheap[teO] + np.where(esc, lO_str[teO], 0)).sum()      # bo-N parallel -> 1 cheap-pass latency
            base_cost += cO_str[teO].sum(); base_lat += lO_str[teO].sum(); escO = esc.mean()
        nt = nMt + nOt
        agg["acc"].append(corr / nt)
        agg["strong_acc"].append(base_corr / nt)
        agg["cheap_acc"].append(cheap_only_corr / nt)
        agg["flops_frac"].append(cost / base_cost)
        agg["flops_frac_shared"].append(costS / base_cost)
        agg["lat_frac"].append(lat / base_lat)
        agg["escM"].append(escM); agg["escO"].append(escO)
        agg["tm"].append(tm); agg["to"].append(to)

    # ---- report ----
    print(f"\n############  {name}  {'(excl MedXpert)' if exclude_medxpert else ''}  ############")
    print(f"  detector on MedEvalKit items: {dict(catch_m)}   (open excluded from MCQ pool: {dict(excl)})")
    print(f"  pool composition: MCQ/closed n={nM}  +  open n={nO}   ->  total pooled n={nM+nO}")
    fmt_break = Counter(M["fmt"]) if nM else Counter()
    print(f"    MCQ-pool formats: {dict(fmt_break)}   open-pool datasets: {dict(Counter(O['bench'])) if nO else {}}")
    a = lambda k: float(np.mean(agg[k]))
    print(f"  --- pooled (honest 50/50, {len(list(seeds))} seeds) ---")
    print(f"    always-strong  : acc={a('strong_acc'):.4f}   FLOPs=1.000  (baseline)")
    print(f"    always-cheap   : acc={a('cheap_acc'):.4f}   (7B MCQ + verifier-bo8 open, no escalation)")
    print(f"    ROUTER (Method C): acc={a('acc'):.4f}   FLOPs={a('flops_frac'):.3f}"
          f"  (prefix-shared FLOPs={a('flops_frac_shared'):.3f})   latency={a('lat_frac'):.3f}")
    print(f"       -> vs always-strong: acc {a('acc')-a('strong_acc'):+.4f}   FLOPs {(a('flops_frac')-1)*100:+.1f}%"
          f"   (shared {(a('flops_frac_shared')-1)*100:+.1f}%)   latency {(a('lat_frac')-1)*100:+.1f}%")
    print(f"       escalation: MCQ {a('escM')*100:.0f}%   open {a('escO')*100:.0f}%   "
          f"(tau_mcq~{a('tm'):.3f}, tau_open~{a('to'):.3f})")
    print(f"       [latency = amortized-batch latency_s proxy; batch-1 measurement queued as GPU job]")
    return dict(name=name, nM=nM, nO=nO, acc=a("acc"), strong=a("strong_acc"), cheap=a("cheap_acc"),
                flops=a("flops_frac"), flops_shared=a("flops_frac_shared"), lat=a("lat_frac"),
                escM=a("escM"), escO=a("escO"))

def main():
    print("=" * 96)
    print("UNIFIED DETERMINISTIC ROUTER (Method C) — pooled MCQ(margin) + open(verifier) vs always-strong")
    print("=" * 96)
    rows = []
    for name, cfg in FAMILIES.items():
        r = eval_family(name, cfg)
        if r: rows.append(r)
        eval_family(name, cfg, exclude_medxpert=True)
    print("\n" + "=" * 96)
    print(f"  {'family':<14}{'pooled n':>9}{'strong':>8}{'router':>8}{'d-acc':>8}{'FLOPs':>8}{'shared':>8}{'latency':>9}")
    for r in rows:
        print(f"  {r['name']:<14}{r['nM']+r['nO']:>9}{r['strong']:>8.3f}{r['acc']:>8.3f}"
              f"{r['acc']-r['strong']:>+8.3f}{r['flops']:>8.3f}{r['flops_shared']:>8.3f}{r['lat']:>9.3f}")
    os.makedirs(J("results/cascade_methods/artifacts"), exist_ok=True)
    json.dump(rows, open(J("results/cascade_methods/artifacts/unified_router.json"), "w"), indent=1)
    print("\n-> results/cascade_methods/artifacts/unified_router.json")

if __name__ == "__main__":
    main()
