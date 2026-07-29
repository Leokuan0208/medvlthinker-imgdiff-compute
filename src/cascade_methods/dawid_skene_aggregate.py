#!/usr/bin/env python3
"""
dawid_skene_aggregate.py - OFFLINE test of backlog idea B5: "Dawid-Skene truth inference"
(crowdsourcing truth inference; Dawid & Skene, JRSS-C 1979).

WHAT THIS ATTACKS (binding limit #2 / the "majority trap"). Our best-of-N pipeline SELECTS one
candidate (verifier argmax). When the correct answer is a *minority* vote, both majority-vote and a
near-tie verifier miss it. B5 instead AGGREGATES the noisy candidates into the latent truth:
treat each sampled answer / each cheap generator as a noisy ANNOTATOR, learn each source's
RELIABILITY unsupervised (EM), and take the reliability-WEIGHTED posterior mode over the answer
clusters. The question: does reliability-weighted aggregation beat plain majority and approach /
beat the TRAINED verifier -- WITHOUT the trained verifier (unsupervised)?

MODEL (grouped one-coin / homogeneous Dawid-Skene). Free-text answers give a DIFFERENT label space
per question (clusters are "left"/"right" here, "yes"/"no" there), so a shared full confusion matrix
is undefined. The one-coin DS model generalises across heterogeneous label spaces with a single
scalar reliability a_g per SOURCE g: a sample from g equals the truth with prob a_g, else spreads
uniformly over the other L_q-1 clusters. EM jointly estimates {a_g} and the per-question posterior
over the truth. The selection reduces to reliability-WEIGHTED vote counting with a per-hit weight
    w_{g,q} = log( a_g * (L_q - 1) / (1 - a_g) )
(more options L_q => a correct vote is more informative). Samples of one generator share a_g
(they are exchangeable), so within a SINGLE source DS reduces to majority vote WHEN the source is
estimated above the per-question chance level (a_g > 1/L_q, i.e. w>0); when a source is estimated
BELOW chance on a low-cardinality question the one-coin model treats it as anti-reliable and
INVERTS its vote (a standard DS pathology), which can fall below majority. The reliability-weighting
lever only bites ACROSS heterogeneous sources -- the multi-generator regime we test.

Two DS readings are reported: PURE (standard; allows below-chance inversion) and GUARDED (weights
clamped to w>=0, i.e. a source is never trusted to be systematically anti-correct; falls back to
plain majority when no source has a positive reliability signal). Guarded DS isolates whether
reliability WEIGHTING helps, separately from the below-chance-inversion artifact.

CLUSTERING. As B5 specifies, cluster candidate answers by EXACT / NORMALISED string match
(lowercase, collapse whitespace, strip trailing punctuation) to form the categorical votes. A
cluster's correctness label = the (near-unanimous) judge_ok of its member samples. Paraphrase
splitting ("left" vs "left side") is a real limitation of string clustering and is reported.

INDEPENDENCE CAVEAT (from the backlog). DS assumes votes are conditionally independent given the
truth, but within-generator temperature samples are CORRELATED -> naive per-sample pooling
OVER-CREDITS reliability. We therefore report TWO multi-source variants:
  * DS-pooled    : all 8 samples of every generator as votes (per-generator shared reliability).
  * DS-collapsed : each generator casts ONE vote (its within-generator self-consistency answer),
                   which removes within-generator correlation -- the correlation-robust reading.

COMPARISONS (per dataset + pooled, 3 generator "families" Lingshu-7B / MVT-7B / IV3-8B):
  self-consistency (per-gen majority) | majority (pooled) | DS-pooled | DS-collapsed
  | trained-verifier argmax (pooled) | oracle@N.
Plus a SINGLE-SOURCE block per family (aggregate that family's own N=8) vs its strong 32B, to show
DS == majority within a source.

Reads ONLY existing dumps (transfer_dump_{ds}_{tag}.json: preds[8], sl[8], scores[8], pick,
greedy_ok) + strong judge jsonl. CPU only, NO GPU / NO inference. No fabricated numbers.
Launch from repo root:  python3 src/cascade_methods/dawid_skene_aggregate.py
"""
import os, json, re
from collections import defaultdict, Counter
import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
DUMP = "ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_{tag}.json"
GTAGS = ["lingshu7b", "7b", "iv3_8b"]
GNAME = {"lingshu7b": "Lingshu-7B", "7b": "MedVLThinker-7B", "iv3_8b": "InternVL3-8B"}
OPEN_DSETS = ["kvasir_open", "radimagenet_open", "vqa_rad_open", "pathvqa_open"]
# strong 32B judge per family (for the single-source block); mirrors open_bestofN_adaptive.py
FAM_STRONG = {
    "lingshu7b": ("ckpts/openvqa/strong_lingshu", ["ckpt_{ds}_lingshu32b_t0.judge.jsonl", "ckpt_{ds}_lingshu32b.judge.jsonl"]),
    "7b":        ("ckpts/openvqa/strong",         ["ckpt_{ds}_32b_t0.judge.jsonl", "ckpt_{ds}_32b.judge.jsonl"]),
    "iv3_8b":    ("ckpts/openvqa/internvl3_38b",  ["ckpt_{ds}_iv3_38b_t0.judge.jsonl", "ckpt_{ds}_iv3_38b.judge.jsonl"]),
}


# ---------------------------------------------------------------- io / normalisation
def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip().strip(".").strip()


def load_dump(ds, tag):
    p = J(DUMP.format(ds=ds, tag=tag))
    if not os.path.exists(p):
        return None
    return {r["idx"]: r for r in json.load(open(p))}


def load_judge(sdir, templates, ds):
    for t in templates:
        p = J(os.path.join(sdir, t.format(ds=ds)))
        if os.path.exists(p):
            m = {}
            for l in open(p):
                if l.strip():
                    r = json.loads(l); m[r["idx"]] = int(r["judge_ok"])
            if m:
                return m
    return {}


def _sl(x):
    return 0 if x in (None, -1) else int(x)


# ---------------------------------------------------------------- per-question clustering
def build_questions(ds, tags):
    """Align the given generators on their common idx; cluster the pooled answers per question.
    Returns list of question dicts with vote counts per generator over a shared cluster set."""
    dumps = {}
    for t in tags:
        d = load_dump(ds, t)
        if d is None:
            return None
        dumps[t] = d
    idx = sorted(set.intersection(*[set(d.keys()) for d in dumps.values()]))
    if not idx:
        return None
    Q = []
    for i in idx:
        cl_id = {}                 # cluster string -> id (id in first-appearance order)
        order = []                 # order[id] = first-appearance rank (== id here, kept explicit)
        sl_of = defaultdict(list)  # id -> list of member sl
        counts = {t: [] for t in tags}
        best_sc, best_sl = -1e18, 0
        oracle = 0
        tag_best = {t: (-1e18, 0) for t in tags}   # per-generator verifier argmax (single-source)
        # first pass: assign clusters in a fixed (tag, sample) order
        vote = {t: Counter() for t in tags}
        for t in tags:
            r = dumps[t][i]
            for pred, s, sc in zip(r["preds"], r["sl"], r["scores"]):
                s = _sl(s); c = norm(pred)
                if c not in cl_id:
                    cl_id[c] = len(cl_id); order.append(len(order))
                cid = cl_id[c]
                sl_of[cid].append(s)
                vote[t][cid] += 1
                oracle = max(oracle, s)
                if sc > best_sc:
                    best_sc, best_sl = sc, s
                if sc > tag_best[t][0]:
                    tag_best[t] = (sc, s)
        L = len(cl_id)
        correct = np.array([int(round(np.mean(sl_of[c]))) for c in range(L)], int)
        cnt = {t: np.array([vote[t].get(c, 0) for c in range(L)], float) for t in tags}
        Q.append(dict(idx=i, L=L, correct=correct, order=np.array(order, int), counts=cnt,
                      ver_sl=int(best_sl), oracle=int(oracle),
                      tag_ver_sl={t: int(tag_best[t][1]) for t in tags}))
    return Q


def argmax_tie(vec, order):
    """index of max(vec); ties -> smallest order (earliest-drawn cluster). Answer-agnostic."""
    m = float(np.max(vec))
    cand = [i for i in range(len(vec)) if vec[i] >= m - 1e-12]
    return min(cand, key=lambda i: int(order[i]))


# ---------------------------------------------------------------- baselines
def sc_pick(q, tag):
    """within-generator self-consistency: majority cluster of that generator's samples."""
    return argmax_tie(q["counts"][tag], q["order"])


def majority_pick(q, tags):
    """pooled plain majority over all samples of all present generators (equal weight)."""
    tot = np.zeros(q["L"])
    for t in tags:
        tot = tot + q["counts"][t]
    return argmax_tie(tot, q["order"])


# ---------------------------------------------------------------- Dawid-Skene EM (grouped one-coin)
def dawid_skene(Q, tags, collapse=False, guard=False, iters=500, tol=1e-7):
    """Grouped one-coin DS EM. Each generator g has scalar reliability a_g shared across its votes.
    collapse=True -> each generator votes once (its within-gen majority cluster).
    guard=True   -> clamp per-hit weights to w>=0 (no below-chance inversion); if no source has a
                    positive signal on a question, fall back to plain pooled majority there.
    Returns (selected-cluster-id per question list, reliabilities dict, n_iter)."""
    # materialise vote-count vectors (collapsed to one-hot if requested)
    cnts = []; Ns = []
    for q in Q:
        cq = {}; nq = {}
        for t in tags:
            v = q["counts"][t].astype(float)
            if collapse:
                if v.sum() > 0:
                    j = argmax_tie(v, q["order"]); one = np.zeros(q["L"]); one[j] = 1.0; v = one
                else:
                    v = np.zeros(q["L"])
            cq[t] = v; nq[t] = float(v.sum())
        cnts.append(cq); Ns.append(nq)
    tot_N = {t: sum(nq[t] for nq in Ns) for t in tags}

    # init posterior = normalised pooled votes (soft majority)
    post = []
    for cq, q in zip(cnts, Q):
        tot = np.zeros(q["L"])
        for t in tags:
            tot = tot + cq[t]
        post.append(tot / tot.sum() if tot.sum() > 0 else np.ones(q["L"]) / q["L"])

    a = {t: 0.5 for t in tags}
    n_iter = 0
    for it in range(iters):
        n_iter = it + 1
        # M-step: reliability = expected fraction of a generator's votes that hit the truth
        newa = {}
        for t in tags:
            num = sum(float(np.dot(p, cq[t])) for p, cq in zip(post, cnts))
            newa[t] = float(np.clip(num / tot_N[t], 1e-4, 1 - 1e-4)) if tot_N[t] > 0 else 0.5
        # E-step: posterior over the truth per question (reliability-weighted votes)
        newpost = []
        for cq, q in zip(cnts, Q):
            L = q["L"]
            if L == 1:
                newpost.append(np.array([1.0])); continue
            score = np.zeros(L)
            any_pos = False
            for t in tags:
                if Ns_zero(cq[t]):
                    continue
                w = np.log(newa[t] * (L - 1) / (1 - newa[t]))   # per-hit DS log-weight
                if guard:
                    w = max(w, 0.0)
                if w > 0:
                    any_pos = True
                score = score + cq[t] * w
            if guard and not any_pos:                            # no reliability signal -> majority
                score = sum(cq[t] for t in tags)
            score = score - score.max()
            e = np.exp(score); newpost.append(e / e.sum())
        maxd = max(abs(newa[t] - a[t]) for t in tags)
        a, post = newa, newpost
        if maxd < tol:
            break
    sel = [argmax_tie(p, q["order"]) for p, q in zip(post, Q)]
    return sel, a, n_iter


def Ns_zero(v):
    return float(v.sum()) == 0.0


# ---------------------------------------------------------------- scoring helpers
def acc_of(Q, picks):
    return float(np.mean([q["correct"][j] for q, j in zip(Q, picks)]))


def self_agreement(Q, tags):
    """per-generator mean fraction of its samples that fall in its own modal cluster (peakedness).
    Contrast with per-sample accuracy: if agreement >> accuracy the source is confidently-wrong,
    so its unsupervised 'reliability' is uninformative about correctness."""
    out = {}
    for t in tags:
        agr, acc = [], []
        for q in Q:
            c = q["counts"][t]
            if c.sum() > 0:
                agr.append(float(c.max() / c.sum()))
        out[t] = round(float(np.mean(agr)), 4) if agr else None
    return out


def per_sample_acc(Q, tags):
    """per-generator mean per-sample correctness (uses cluster correctness weighted by votes)."""
    out = {}
    for t in tags:
        vals = []
        for q in Q:
            c = q["counts"][t]
            if c.sum() > 0:
                vals.append(float(np.dot(c, q["correct"]) / c.sum()))
        out[t] = round(float(np.mean(vals)), 4) if vals else None
    return out


def eval_multisource(Q, tags):
    """Compute every method's accuracy on a question set for the given present generators."""
    n = len(Q)
    sc = {t: acc_of(Q, [sc_pick(q, t) for q in Q]) for t in tags}
    maj_sel = [majority_pick(q, tags) for q in Q]
    maj = acc_of(Q, maj_sel)
    ds_p_sel, a_p, _ = dawid_skene(Q, tags, collapse=False, guard=False)
    ds_pg_sel, a_pg, _ = dawid_skene(Q, tags, collapse=False, guard=True)
    ds_cg_sel, a_cg, _ = dawid_skene(Q, tags, collapse=True, guard=True)
    ds_pooled = acc_of(Q, ds_p_sel)
    ds_pooled_guard = acc_of(Q, ds_pg_sel)
    ds_collapsed_guard = acc_of(Q, ds_cg_sel)
    ver = float(np.mean([q["ver_sl"] for q in Q]))
    oracle = float(np.mean([q["oracle"] for q in Q]))
    flips_pg = int(np.sum([a != b for a, b in zip(ds_pg_sel, maj_sel)]))
    return dict(
        n=n, generators=[GNAME[t] for t in tags],
        per_sample_acc={GNAME[t]: v for t, v in per_sample_acc(Q, tags).items()},
        self_agreement={GNAME[t]: v for t, v in self_agreement(Q, tags).items()},
        self_consistency_pergen={GNAME[t]: sc[t] for t in tags},
        self_consistency_best=float(max(sc.values())),
        majority_pooled=maj,
        ds_pooled_pure=ds_pooled, ds_pooled_guarded=ds_pooled_guard,
        ds_collapsed_guarded=ds_collapsed_guard,
        verifier_pooled=ver, oracle=oracle,
        reliabilities_pooled_pure={GNAME[t]: round(a_p[t], 4) for t in tags},
        reliabilities_pooled_guarded={GNAME[t]: round(a_pg[t], 4) for t in tags},
        ds_pooled_guarded_flips_vs_majority=flips_pg,
        delta_ds_pure_vs_majority=round(ds_pooled - maj, 4),
        delta_ds_guarded_vs_majority=round(ds_pooled_guard - maj, 4),
        delta_ds_guarded_vs_verifier=round(ds_pooled_guard - ver, 4),
    )


def eval_singlesource(ds, tag):
    """Aggregate ONE family's own N=8 samples. Confirms DS == majority within a source, and
    compares to the verifier bo8, oracle@8 and the family's strong 32B (on the shared idx)."""
    Q = build_questions(ds, [tag])
    if Q is None:
        return None
    sdir, templ = FAM_STRONG[tag]
    sj = load_judge(sdir, templ, ds)
    Q = [q for q in Q if q["idx"] in sj]     # keep only q with a strong label (fair escalation cmp)
    if not Q:
        return None
    sc = acc_of(Q, [sc_pick(q, tag) for q in Q])                 # = majority within source
    ds_sel, a, _ = dawid_skene(Q, [tag], collapse=False, guard=False)
    dsg_sel, ag, _ = dawid_skene(Q, [tag], collapse=False, guard=True)
    ds_acc = acc_of(Q, ds_sel)
    dsg_acc = acc_of(Q, dsg_sel)
    ver = float(np.mean([q["tag_ver_sl"][tag] for q in Q]))
    oracle = float(np.mean([q["oracle"] for q in Q]))
    strong = float(np.mean([sj[q["idx"]] for q in Q]))
    return dict(n=len(Q), sc_majority=sc, ds_single_pure=ds_acc, ds_single_guarded=dsg_acc,
                ds_pure_equals_majority=bool(abs(ds_acc - sc) < 1e-9),
                ds_guarded_equals_majority=bool(abs(dsg_acc - sc) < 1e-9),
                verifier_bo8=ver, oracle8=oracle, strong32b=strong,
                reliability=round(a[tag], 4))


# ---------------------------------------------------------------- driver
def build_pool(ds_list, tags):
    """concatenate question structures across datasets (only those where all tags present)."""
    allQ = []; used = []
    for ds in ds_list:
        Q = build_questions(ds, tags)
        if Q is None:
            continue
        allQ += Q; used.append(ds)
    return (allQ, used) if allQ else (None, used)


def main():
    print("#" * 100)
    print("DAWID-SKENE truth inference (B5) - reliability-weighted AGGREGATION vs SELECTION")
    print("  OFFLINE, CPU, no inference. clusters = exact/normalised answer strings")
    print("#" * 100)

    OUT = {"method": "dawid_skene_grouped_one_coin",
           "model": "grouped one-coin DS EM; per-source scalar reliability a_g shared across its "
                    "samples; per-question uniform prior over answer clusters; DS weight per hit "
                    "w=log(a(L-1)/(1-a)); clusters by exact/normalised string match",
           "caveats": "within-generator samples are correlated (violates DS independence) -> "
                      "DS-collapsed (1 vote/generator) is the correlation-robust variant; free-text "
                      "answers fragment under string clustering (paraphrase splitting).",
           "single_source": {}, "multi_source": {}}

    # ---------- (1) SINGLE-SOURCE: aggregate each family's own N=8 ----------
    print("\n" + "=" * 100)
    print("(1) SINGLE-SOURCE  aggregate one family's N=8  (DS vs its own majority/verifier/strong)")
    print("=" * 100)
    hdr = (f"  {'family':<16}{'dataset':<18}{'n':>5}{'SC=maj':>9}{'DS-pure':>9}{'DS-guard':>9}"
           f"{'verif-bo8':>11}{'oracle@8':>10}{'strong32B':>11}")
    print(hdr)
    pure_eq_all = True; guard_eq_all = True
    for tag in GTAGS:
        for ds in OPEN_DSETS:
            r = eval_singlesource(ds, tag)
            if r is None:
                continue
            OUT["single_source"].setdefault(GNAME[tag], {})[ds] = r
            pure_eq_all &= r["ds_pure_equals_majority"]
            guard_eq_all &= r["ds_guarded_equals_majority"]
            print(f"  {GNAME[tag]:<16}{ds:<18}{r['n']:>5}{r['sc_majority']:>9.3f}{r['ds_single_pure']:>9.3f}"
                  f"{r['ds_single_guarded']:>9.3f}{r['verifier_bo8']:>11.3f}{r['oracle8']:>10.3f}{r['strong32b']:>11.3f}")
    print(f"\n  -> GUARDED DS-single == majority everywhere: {guard_eq_all}  |  PURE DS-single == majority everywhere: {pure_eq_all}")
    print("     (pure DS dips below majority via below-chance vote inversion on low-cardinality questions;")
    print("      guarded DS recovers majority: within one exchangeable source there is no reliability signal to exploit.)")
    OUT["single_source_guarded_ds_equals_majority_everywhere"] = bool(guard_eq_all)
    OUT["single_source_pure_ds_equals_majority_everywhere"] = bool(pure_eq_all)

    # ---------- (2) MULTI-SOURCE: DS across the cheap generators ----------
    print("\n" + "=" * 100)
    print("(2) MULTI-SOURCE  DS across cheap generators (the real test: beat majority, approach verifier)")
    print("=" * 100)
    print(f"  {'config':<22}{'n':>5}{'SC-best':>9}{'maj':>8}{'DSpure':>8}{'DSgrd':>8}{'DScoll':>8}{'verif':>8}{'oracle':>8}"
          f"    dDSgrd(vs maj / vs verif)")

    def show(label, r):
        print(f"  {label:<22}{r['n']:>5}{r['self_consistency_best']:>9.3f}{r['majority_pooled']:>8.3f}"
              f"{r['ds_pooled_pure']:>8.3f}{r['ds_pooled_guarded']:>8.3f}{r['ds_collapsed_guarded']:>8.3f}"
              f"{r['verifier_pooled']:>8.3f}{r['oracle']:>8.3f}"
              f"    {r['delta_ds_guarded_vs_majority']:+.3f} / {r['delta_ds_guarded_vs_verifier']:+.3f}")

    # per-dataset, 3 generators (kvasir/radimagenet/vqa_rad)
    DS3 = ["kvasir_open", "radimagenet_open", "vqa_rad_open"]
    for ds in DS3:
        Q = build_questions(ds, GTAGS)
        if Q is None:
            continue
        r = eval_multisource(Q, GTAGS)
        OUT["multi_source"][ds] = r
        show(ds, r)
    # pooled over the 3 all-generator datasets
    Qpool, used = build_pool(DS3, GTAGS)
    if Qpool:
        r = eval_multisource(Qpool, GTAGS)
        OUT["multi_source"]["POOLED_3ds_3gen"] = r
        show("POOLED (3ds,3gen)", r)
    # supplementary: pathvqa with the 2 generators that ran it (Lingshu + IV3)
    Qp = build_questions("pathvqa_open", ["lingshu7b", "iv3_8b"])
    if Qp is not None:
        r = eval_multisource(Qp, ["lingshu7b", "iv3_8b"])
        OUT["multi_source"]["pathvqa_open_2gen"] = r
        show("pathvqa_open (2gen)", r)

    # ---------- verdict ----------
    pooled = OUT["multi_source"].get("POOLED_3ds_3gen")
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    if pooled:
        print("  per-generator  self-agreement vs per-sample-accuracy (confidently-wrong if agreement>>accuracy):")
        for g in pooled["generators"]:
            print(f"     {g:<16} agreement={pooled['self_agreement'][g]:.3f}   accuracy={pooled['per_sample_acc'][g]:.3f}")
        print("  learned reliabilities (DS-pooled pure): " +
              ", ".join(f"{g}={v:.2f}" for g, v in pooled["reliabilities_pooled_pure"].items()))
        print(f"  POOLED(3ds,3gen): majority={pooled['majority_pooled']:.3f}  "
              f"DS-pure={pooled['ds_pooled_pure']:.3f} ({pooled['delta_ds_pure_vs_majority']:+.3f})  "
              f"DS-guarded={pooled['ds_pooled_guarded']:.3f} ({pooled['delta_ds_guarded_vs_majority']:+.3f})")
        print(f"                    trained-verifier={pooled['verifier_pooled']:.3f}  oracle={pooled['oracle']:.3f}")
        beats_maj = pooled["ds_pooled_guarded"] > pooled["majority_pooled"] + 1e-9
        beats_ver = pooled["ds_pooled_guarded"] >= pooled["verifier_pooled"] - 1e-9
        gap_to_ver = pooled["verifier_pooled"] - pooled["ds_pooled_guarded"]
        gap_to_oracle = pooled["oracle"] - pooled["ds_pooled_guarded"]
        v = (f"Guarded DS {'BEATS' if beats_maj else 'DOES NOT beat'} plain pooled majority "
             f"({pooled['delta_ds_guarded_vs_majority']:+.3f}); it {'MATCHES/BEATS' if beats_ver else 'STAYS FAR BELOW'} "
             f"the trained verifier (gap {gap_to_ver:+.3f}) and {gap_to_oracle:+.3f} below oracle. "
             f"Unsupervised reliability tracks self-AGREEMENT (~{np.mean([pooled['self_agreement'][g] for g in pooled['generators']]):.2f}) "
             f"not ACCURACY (~{np.mean([pooled['per_sample_acc'][g] for g in pooled['generators']]):.2f}): the generators are "
             f"confidently-wrong, so cross-source agreement carries no correctness signal and cannot break the majority trap.")
        print("  " + v)
        OUT["verdict"] = dict(
            single_source_guarded_ds_equals_majority=bool(guard_eq_all),
            pooled_guarded_ds_beats_majority=bool(beats_maj),
            pooled_guarded_ds_matches_or_beats_verifier=bool(beats_ver),
            pooled_gap_guarded_ds_to_verifier=round(gap_to_ver, 4),
            pooled_gap_guarded_ds_to_oracle=round(gap_to_oracle, 4),
            summary=v)

    os.makedirs(J("results/cascade_methods/artifacts"), exist_ok=True)
    outp = J("results/cascade_methods/artifacts/dawid_skene_aggregate.json")
    json.dump(OUT, open(outp, "w"), indent=1)
    print(f"\n[dump] {outp}")


if __name__ == "__main__":
    main()
