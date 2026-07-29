#!/usr/bin/env python3
"""
active_comparison_verifier.py  --  OFFLINE test of backlog idea C9:
"Information-directed active-comparison verifier (TrueSkill mu,sigma / IDS)".

WHAT C9 IS (backlog METHOD_IDEAS_BACKLOG.md, overall #3):
  Instead of picking the best-of-N candidate by a single POINTWISE verifier argmax (current
  method, ~74-82% selection efficiency on within-question near-ties = binding limit #2), give
  each candidate a TrueSkill-style rating (mu, sigma); at each step run the MOST INFORMATIVE
  pairwise comparison (information-directed / max match-quality among the plausibly-best
  candidates), update the ratings, and STOP once the top candidate's win-probability is
  confident.  Selection = top-mu candidate.  Attacks limit #2 (near-tie selection) AND cuts
  verify calls vs a blind knockout/round-robin (both-axes on the verify side).

HARD OFFLINE CONSTRAINT: no GPU / no vLLM / no eval job. Reads ONLY existing per-sample dumps
  (ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_{vtag}.json): per question we have
  sl[N]   = per-candidate correctness (judge_ok, 0/1),
  scores[N] = per-candidate POINTWISE trained-verifier P(Yes),
  preds[N]  = the N candidate answer strings,
  greedy_ok = SC/greedy correctness.

>>> CRITICAL HONESTY NOTE — SIMULATED PAIRWISE PREFERENCES <<<
  The repo has NO pairwise verifier dumps (no "(img,q,ansA,ansB)->A/B" verdicts). Only pointwise
  P(Yes) scores exist. We therefore SIMULATE the outcome of a pairwise comparison from the
  pointwise scores with the Bradley-Terry model that is EXACTLY CONSISTENT with the pointwise
  verifier:  P(i beats j) = sigmoid( logit(s_i) - logit(s_j) )   (candidate "skill" = logit P(Yes)).
  Consequence, stated up front and confirmed by the Copeland/round-robin ceiling below: because
  the simulated pairwise preference is a MONOTONE function of the pointwise score, any pairwise
  aggregation (knockout, round-robin/Copeland, TrueSkill) ranks candidates by their pointwise
  score in expectation, so its SELECTION-ACCURACY CEILING EQUALS pointwise-argmax. Under this
  simulation C9 CANNOT beat pointwise on accuracy -- it can only (a) match it and (b) trade
  verify COST vs a full pairwise pass. To actually TEST whether active comparison beats pointwise
  on accuracy you need REAL pairwise verifier verdicts (a small GPU pass, sized at the end).

METRICS (per dataset + pooled, 3 families):
  sel_acc  = mean_q  sl[selected]                              (overall selection accuracy)
  sel_eff  = mean_{q: max(sl)==1} sl[selected]  = P(pick correct | a correct candidate exists)
             -- THIS is the 74-82% "selection ceiling" metric that limit #2 refers to.
  cost     = mean number of pairwise comparisons used (pointwise-argmax uses 0; it only reads
             the N scores it already has).
  Also: oracle@N (upper bound), SC/greedy (context), and a NEAR-TIE subset (top-2 score gap<0.1)
  where selection actually matters.

METHODS COMPARED:
  pointwise-argmax (DEPLOYED)  -- argmax(scores);                       cost 0
  blind knockout (B1)          -- index-seeded single elimination;      cost N-1
  round-robin + Copeland       -- all pairs, pick max wins;             cost N(N-1)/2  (=28 for N=8)
  active-IDS TrueSkill (C9)    -- verifier-seeded ratings + info-directed comparisons + sigma-stop
  oracle@N                     -- max(sl); ceiling.

Pairwise outcomes are drawn stochastically (Bernoulli) with SEEDED common-random-numbers, averaged
over several seeds (mean +/- std reported). A DETERMINISTIC Copeland is also reported as the clean
"noise-free pairwise ceiling".

Held-out: the ONLY fitted knob is the C9 stop-confidence p_conf. We fit it on a 50% calibration
split (smallest p_conf whose active-IDS sel_eff matches round-robin within tol) and report the
held-out 50% result + the full frontier for transparency.

Launch from repo root:  python3 src/cascade_methods/active_comparison_verifier.py
Writes: results/cascade_methods/artifacts/active_comparison_verifier.json
"""
import os, json, math
import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
ADAPTER = "ckpts/train/lora_verifier_pooled4"
OPEN_DSETS = ["vqa_rad_open", "pathvqa_open", "kvasir_open", "radimagenet_open"]
# 3 families (same wiring as open_bestofN_adaptive.py). We only need the candidate dumps for
# SELECTION; strong-model judge labels are not required (selection is candidate-internal).
FAM = {
    "Lingshu":      dict(vtag="lingshu7b"),
    "MedVLThinker": dict(vtag="7b"),
    "InternVL3":    dict(vtag="iv3_8b"),
}

# ---- pairwise-simulation + TrueSkill constants ------------------------------------------------
EPS      = 1e-3      # clamp for logit(P(Yes))
BETA     = 1.0       # BT temperature (=1 => pairwise preference EXACTLY consistent with pointwise)
SIGMA0   = 0.60      # TrueSkill prior std (logit units) on the verifier-seeded rating
BETA_TS  = 0.50      # TrueSkill performance noise
NEARTIE  = 0.10      # top-2 pointwise-score gap below which a question is "contested"
MC       = 256       # Monte-Carlo samples for P(best)
SEEDS    = [0, 1, 2] # comparison-outcome RNG seeds (report mean +/- std)
PCONF_GRID = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]


def logit(s):
    s = min(max(float(s), EPS), 1.0 - EPS)
    return math.log(s / (1.0 - s))


# ---- REAL pairwise verdicts (Experiment 2) -----------------------------------------------------
# When _PF is set to a callable pf(i,j)->P(i beats j), the pick functions below use it INSTEAD of the
# Bradley-Terry preference simulated from the pointwise score. This is how we drive the SAME active-
# comparison machinery with REAL pairwise verifier verdicts (pairwise_verifier_score.py). _REAL maps
# (ds, idx) -> {(i,j): p_i_gt_j}. Default None => original simulated behavior is byte-for-byte preserved.
_PF = None
_REAL = None


def make_pf(row):
    """Per-row pairwise-probability function. Real verdict if available (either orientation), else the
    pointwise-consistent BT sim (also the fallback for pairs with no verdict, e.g. identical strings)."""
    sc = row["sc"]; z = [logit(s) for s in sc]
    key = (row.get("ds"), row.get("idx"))
    vd = _REAL.get(key) if _REAL is not None else None
    if not vd:
        return lambda i, j: 1.0 / (1.0 + math.exp(-BETA * (z[i] - z[j])))
    def pf(i, j):
        if (i, j) in vd: return min(max(vd[(i, j)], 1e-6), 1 - 1e-6)
        if (j, i) in vd: return min(max(1.0 - vd[(j, i)], 1e-6), 1 - 1e-6)
        return 1.0 / (1.0 + math.exp(-BETA * (z[i] - z[j])))
    return pf


# ---------------------------------------------------------------- data loading (dumps only)
def load_family(vtag):
    per_ds = {}
    for ds in OPEN_DSETS:
        dp = J(os.path.join(ADAPTER, f"transfer_dump_{ds}_{vtag}.json"))
        if not os.path.exists(dp):
            continue
        rows = []
        for r in json.load(open(dp)):
            sl = [0 if x in (None, -1) else int(x) for x in r["sl"]]
            sc = list(r["scores"])
            n = min(len(sl), len(sc))
            if n < 2:
                continue
            rows.append(dict(sl=sl[:n], sc=sc[:n], greedy_ok=int(r.get("greedy_ok", 0))))
        if rows:
            per_ds[ds] = rows
    return per_ds


# ---------------------------------------------------------------- pairwise outcome simulator
def p_i_beats_j(z, i, j):
    """Bradley-Terry preference consistent with the pointwise verifier: sigmoid(z_i - z_j)."""
    return 1.0 / (1.0 + math.exp(-BETA * (z[i] - z[j])))


# ---------------------------------------------------------------- TrueSkill 2-player update
def _v_w(t):
    # win factors (no-draw TrueSkill); Phi/phi with a numerically safe tail
    from math import erf, sqrt, pi, exp
    Phi = 0.5 * (1.0 + erf(t / sqrt(2.0)))
    phi = exp(-0.5 * t * t) / sqrt(2.0 * pi)
    if Phi < 1e-12:               # deep tail: v ~ -t
        v = -t
    else:
        v = phi / Phi
    w = v * (v + t)
    return v, w


def ts_update(mu, s2, w, l):
    """TrueSkill update for outcome winner=w, loser=l (indices). mu, s2 in-place arrays."""
    c2 = 2.0 * BETA_TS * BETA_TS + s2[w] + s2[l]
    c = math.sqrt(c2)
    t = (mu[w] - mu[l]) / c
    v, wf = _v_w(t)
    mu[w] += (s2[w] / c) * v
    mu[l] -= (s2[l] / c) * v
    s2[w] *= (1.0 - (s2[w] / c2) * wf)
    s2[l] *= (1.0 - (s2[l] / c2) * wf)


# ---------------------------------------------------------------- P(best) via common-rand MC
def p_best(mu, s2, Z):
    """Z is a fixed (MC, N) standard-normal matrix (common random numbers)."""
    samp = mu[None, :] + np.sqrt(s2)[None, :] * Z           # (MC, N)
    win = np.argmax(samp, axis=1)
    pb = np.bincount(win, minlength=len(mu)).astype(float) / Z.shape[0]
    return pb


# ---------------------------------------------------------------- the C9 active-IDS trajectory
def active_ids_trajectory(sc, seed):
    """Verifier-seeded TrueSkill + information-directed comparisons.
    Returns a trajectory list of (n_comparisons_used, leader_idx, max_pbest) recorded BEFORE any
    comparison (step 0 = the pure pointwise prior) and after each comparison, until max_pbest
    >= 0.995 or the round-robin budget is spent. Sweeping the stop-threshold p_conf over this
    single trajectory yields the whole accuracy<->cost frontier for free."""
    n = len(sc)
    z = np.array([logit(s) for s in sc], float)
    mu = z.copy()                                    # verifier score seeds the mean
    s2 = np.full(n, SIGMA0 * SIGMA0, float)          # prior variance
    rng = np.random.default_rng(1000 + seed)
    Z = rng.standard_normal((MC, n))                 # common random numbers for P(best)
    orng = np.random.default_rng(7000 + seed)        # comparison-outcome stream

    traj = []
    pb = p_best(mu, s2, Z)
    leader = int(np.argmax(mu))
    traj.append((0, leader, float(pb.max())))

    budget = n * (n - 1) // 2
    seen = set()
    for step in range(budget):
        if pb.max() >= 0.995:
            break
        # information-directed: restrict to plausibly-best candidates (P(best) floor) + top-2 by mu,
        # then pick the pair of max TrueSkill match-quality (draw prob): closest mu, highest sigma.
        pool = set(np.where(pb >= 0.02)[0].tolist())
        pool.update(np.argsort(mu)[-2:].tolist())
        pool = sorted(pool)
        best_q, best_pair = -1.0, None
        for a_i in range(len(pool)):
            for b_i in range(a_i + 1, len(pool)):
                i, jj = pool[a_i], pool[b_i]
                c2 = 2.0 * BETA_TS * BETA_TS + s2[i] + s2[jj]
                q = math.sqrt((2.0 * BETA_TS * BETA_TS) / c2) * math.exp(-((mu[i] - mu[jj]) ** 2) / (2.0 * c2))
                # tiny penalty for re-running an already-seen pair, so info keeps flowing
                if (i, jj) in seen:
                    q *= 0.25
                if q > best_q:
                    best_q, best_pair = q, (i, jj)
        if best_pair is None:
            break
        i, jj = best_pair
        seen.add((i, jj))
        # comparison outcome (Bernoulli): REAL verdict via _PF if set, else BT-sim from pointwise
        p = _PF(i, jj) if _PF is not None else 1.0 / (1.0 + math.exp(-BETA * (z[i] - z[jj])))
        if orng.random() < p:
            ts_update(mu, s2, i, jj)
        else:
            ts_update(mu, s2, jj, i)
        pb = p_best(mu, s2, Z)
        leader = int(np.argmax(mu))
        traj.append((step + 1, leader, float(pb.max())))
    return traj


def get_traj(row, seed):
    """Cache the (p_conf-independent) active-IDS trajectory on the row dict, keyed by seed."""
    cache = row.setdefault("_traj", {})
    if seed not in cache:
        cache[seed] = active_ids_trajectory(row["sc"], seed)
    return cache[seed]


def select_at_pconf(traj, p_conf):
    """First trajectory point whose max_pbest >= p_conf (else the last point). -> (leader, cost)."""
    for (ncmp, leader, mpb) in traj:
        if mpb >= p_conf:
            return leader, ncmp
    ncmp, leader, _ = traj[-1]
    return leader, ncmp


# ---------------------------------------------------------------- blind baselines
def knockout_pick(sc, seed):
    """Index-seeded single-elimination bracket; N-1 comparisons; stochastic BT outcomes."""
    n = len(sc)
    z = [logit(s) for s in sc]
    rng = np.random.default_rng(9000 + seed)
    alive = list(range(n))
    ncmp = 0
    while len(alive) > 1:
        nxt = []
        for k in range(0, len(alive) - 1, 2):
            i, jj = alive[k], alive[k + 1]
            ncmp += 1
            p = _PF(i, jj) if _PF is not None else 1.0 / (1.0 + math.exp(-BETA * (z[i] - z[jj])))
            nxt.append(i if rng.random() < p else jj)
        if len(alive) % 2 == 1:
            nxt.append(alive[-1])               # bye
        alive = nxt
    return alive[0], ncmp


def copeland_pick_det(sc):
    """Deterministic round-robin Copeland (noise-free pairwise ceiling). max wins; ties->argmax score."""
    n = len(sc)
    z = [logit(s) for s in sc]
    wins = np.zeros(n)
    for i in range(n):
        for jj in range(i + 1, n):
            if _PF is not None:
                p = _PF(i, jj)
                if p > 0.5: wins[i] += 1
                elif p < 0.5: wins[jj] += 1
                else: wins[i] += 0.5; wins[jj] += 0.5
            elif z[i] > z[jj]:
                wins[i] += 1
            elif z[jj] > z[i]:
                wins[jj] += 1
            else:
                wins[i] += 0.5; wins[jj] += 0.5
    # tie-break by pointwise score, then index (matches np.argmax semantics)
    best = max(range(n), key=lambda k: (wins[k], z[k], -k))
    return best, n * (n - 1) // 2


# ---------------------------------------------------------------- per-question evaluation
def eval_rows(rows, method, p_conf=None):
    """Return per-question (correct_selected, correct_present, cost) averaged over SEEDS."""
    global _PF
    accs, effs, effs_den, costs = [], [], [], []
    for r in rows:
        sl, sc = r["sl"], r["sc"]
        _PF = make_pf(r) if _REAL is not None else None   # REAL verdicts drive the pick funcs when loaded
        present = 1 if max(sl) == 1 else 0
        per_seed_ok, per_seed_cost = [], []
        if method == "pointwise":
            j = int(np.argmax(sc)); per_seed_ok = [sl[j]]; per_seed_cost = [0]
        elif method == "oracle":
            per_seed_ok = [max(sl)]; per_seed_cost = [0]
        elif method == "greedy":
            per_seed_ok = [r["greedy_ok"]]; per_seed_cost = [0]
        elif method == "copeland_det":
            j, c = copeland_pick_det(sc); per_seed_ok = [sl[j]]; per_seed_cost = [c]
        elif method == "knockout":
            for s in SEEDS:
                j, c = knockout_pick(sc, s); per_seed_ok.append(sl[j]); per_seed_cost.append(c)
        elif method == "active":
            for s in SEEDS:
                traj = get_traj(r, s)
                leader, c = select_at_pconf(traj, p_conf)
                per_seed_ok.append(sl[leader]); per_seed_cost.append(c)
        else:
            raise ValueError(method)
        ok = float(np.mean(per_seed_ok)); cost = float(np.mean(per_seed_cost))
        accs.append(ok); costs.append(cost)
        if present:
            effs.append(ok); effs_den.append(1)
    sel_acc = float(np.mean(accs))
    sel_eff = float(np.sum(effs) / max(len(effs), 1))
    cost = float(np.mean(costs))
    _PF = None
    return dict(sel_acc=sel_acc, sel_eff=sel_eff, cost=cost, n=len(rows), n_present=len(effs))


def neartie_mask(rows):
    m = []
    for r in rows:
        ss = sorted(r["sc"], reverse=True)
        gap = ss[0] - ss[1] if len(ss) > 1 else ss[0]
        m.append(gap < NEARTIE and max(r["sl"]) == 1)   # contested AND winnable
    return np.array(m, bool)


# ---------------------------------------------------------------- held-out p_conf fit
def fit_pconf(rows_cal, target_eff, tol=0.003):
    """Smallest p_conf whose active-IDS sel_eff on the calibration split >= target - tol
    (i.e. matches the round-robin ceiling as cheaply as possible)."""
    best = PCONF_GRID[-1]
    for pc in PCONF_GRID:
        e = eval_rows(rows_cal, "active", p_conf=pc)["sel_eff"]
        if e >= target_eff - tol:
            best = pc
            break
    return best


# ---------------------------------------------------------------- driver
def run_family(name, per_ds):
    fam = dict(name=name, datasets={}, pooled={})
    all_rows = []
    # OPEN_DSETS order first, then any extra datasets present (e.g. slake_open in the REAL-verdict run)
    order = [d for d in OPEN_DSETS if d in per_ds] + [d for d in per_ds if d not in OPEN_DSETS]
    for ds in order:
        rows = per_ds[ds]
        all_rows.extend(rows)
        fam["datasets"][ds] = eval_block(rows, fit_here=False)
    fam["pooled"] = eval_block(all_rows, fit_here=True)
    fam["n_datasets"] = len(fam["datasets"])
    return fam


def eval_block(rows, fit_here):
    out = {}
    out["greedy"]    = eval_rows(rows, "greedy")
    out["oracle@N"]  = eval_rows(rows, "oracle")
    out["pointwise"] = eval_rows(rows, "pointwise")
    out["knockout"]  = eval_rows(rows, "knockout")
    out["copeland_det"] = eval_rows(rows, "copeland_det")
    # active-IDS frontier (whole accuracy<->cost curve)
    frontier = []
    for pc in PCONF_GRID:
        e = eval_rows(rows, "active", p_conf=pc)
        frontier.append(dict(p_conf=pc, **e))
    out["active_frontier"] = frontier

    # held-out fit of p_conf: split 50/50, fit on calib to match round-robin sel_eff, eval on held-out
    if fit_here and len(rows) >= 40:
        rng = np.random.default_rng(2024)
        perm = rng.permutation(len(rows))
        half = len(rows) // 2
        cal = [rows[i] for i in perm[:half]]
        hld = [rows[i] for i in perm[half:]]
        rr_cal = eval_rows(cal, "copeland_det")["sel_eff"]
        pc_star = fit_pconf(cal, rr_cal)
        held = eval_rows(hld, "active", p_conf=pc_star)
        out["active_heldout"] = dict(p_conf_fit=pc_star, **held)
        out["heldout_pointwise"] = eval_rows(hld, "pointwise")
        out["heldout_copeland_det"] = eval_rows(hld, "copeland_det")
        out["heldout_knockout"] = eval_rows(hld, "knockout")

    # near-tie (contested & winnable) subset selection efficiency
    m = neartie_mask(rows)
    sub = [r for r, keep in zip(rows, m) if keep]
    if sub:
        out["neartie"] = dict(
            n=len(sub),
            pointwise=eval_rows(sub, "pointwise")["sel_acc"],
            knockout=eval_rows(sub, "knockout")["sel_acc"],
            copeland_det=eval_rows(sub, "copeland_det")["sel_acc"],
            active_p90=eval_rows(sub, "active", p_conf=0.90)["sel_acc"],
        )
    return out


def fmt(d):
    return f"acc={d['sel_acc']:.3f} eff={d['sel_eff']:.3f} cost={d['cost']:.1f}"


def main():
    print("=" * 108)
    print("C9  Information-directed active-comparison verifier (TrueSkill mu,sigma / IDS)  --  OFFLINE")
    print("    pairwise preferences SIMULATED from pointwise verifier P(Yes):  P(i>j)=sigmoid(logit s_i - logit s_j)")
    print("=" * 108)
    RESULT = dict(
        note=("Pairwise verdicts SIMULATED from pointwise verifier scores (no pairwise dumps exist). "
              "P(i>j)=sigmoid(logit s_i - logit s_j). Selection-accuracy ceiling of any pairwise "
              "aggregation therefore equals pointwise-argmax; C9 can only match accuracy and trade "
              "verify COST vs a full pairwise pass. Numbers averaged over comparison-seeds "
              f"{SEEDS}."),
        params=dict(BETA=BETA, SIGMA0=SIGMA0, BETA_TS=BETA_TS, MC=MC, NEARTIE=NEARTIE,
                    SEEDS=SEEDS, PCONF_GRID=PCONF_GRID),
        families={},
    )
    for name, cfg in FAM.items():
        per_ds = load_family(cfg["vtag"])
        if not per_ds:
            print(f"\n### {name}: no dumps"); continue
        fam = run_family(name, per_ds)
        RESULT["families"][name] = fam
        p = fam["pooled"]
        print(f"\n############  {name}  (n_datasets={fam['n_datasets']}, n={p['pointwise']['n']})  ############")
        print(f"  SC/greedy      {fmt(p['greedy'])}")
        print(f"  oracle@N       {fmt(p['oracle@N'])}   <- ceiling")
        print(f"  POINTWISE (dep){fmt(p['pointwise'])}   <- current method, 0 comparisons")
        print(f"  knockout (B1)  {fmt(p['knockout'])}")
        print(f"  round-robin RR {fmt(p['copeland_det'])}  <- deterministic Copeland = noise-free pairwise ceiling")
        print(f"  active-IDS frontier (p_conf -> acc/eff/mean-comparisons):")
        for f in p["active_frontier"]:
            print(f"      p_conf={f['p_conf']:.2f}   acc={f['sel_acc']:.3f} eff={f['sel_eff']:.3f} "
                  f"mean_cmp={f['cost']:.2f}")
        if "active_heldout" in p:
            h = p["active_heldout"]; hp = p["heldout_pointwise"]; hr = p["heldout_copeland_det"]
            print(f"  HELD-OUT (p_conf fit on calib half to match RR):  p_conf*={h['p_conf_fit']}")
            print(f"      active-IDS  acc={h['sel_acc']:.3f} eff={h['sel_eff']:.3f} mean_cmp={h['cost']:.2f}")
            print(f"      pointwise   acc={hp['sel_acc']:.3f} eff={hp['sel_eff']:.3f} (0 cmp)")
            print(f"      round-robin acc={hr['sel_acc']:.3f} eff={hr['sel_eff']:.3f} (28 cmp)")
            rr_cost = p['copeland_det']['cost']
            if rr_cost > 0:
                print(f"      => active-IDS matches RR selection using {h['cost']:.1f} vs {rr_cost:.0f} comparisons "
                      f"({(1-h['cost']/rr_cost)*100:.0f}% fewer than full pairwise)")
        if "neartie" in p:
            nt = p["neartie"]
            print(f"  NEAR-TIE subset (top-2 gap<{NEARTIE}, correct present; n={nt['n']}) selection acc:")
            print(f"      pointwise={nt['pointwise']:.3f}  knockout={nt['knockout']:.3f}  "
                  f"round-robin={nt['copeland_det']:.3f}  active@p90={nt['active_p90']:.3f}")

    # -------- pooled-over-families verdict --------
    verdict = build_verdict(RESULT)
    RESULT["verdict"] = verdict
    print("\n" + "=" * 108)
    print("VERDICT")
    print("=" * 108)
    for line in verdict["lines"]:
        print("  " + line)

    outp = J("results/cascade_methods/artifacts/active_comparison_verifier.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(RESULT, open(outp, "w"), indent=1)
    print(f"\n[dump] {outp}")


def build_verdict(RESULT):
    lines = []
    rows = []
    for name, fam in RESULT["families"].items():
        p = fam["pooled"]
        pw = p["pointwise"]["sel_eff"]; rr = p["copeland_det"]["sel_eff"]
        ko = p["knockout"]["sel_eff"]
        # best active frontier eff and its cost
        af = max(p["active_frontier"], key=lambda f: (f["sel_eff"], -f["cost"]))
        rows.append(dict(family=name, n=p["pointwise"]["n"],
                         pointwise_eff=pw, roundrobin_eff=rr, knockout_eff=ko,
                         active_best_eff=af["sel_eff"], active_best_cost=af["cost"],
                         active_best_pconf=af["p_conf"],
                         heldout=p.get("active_heldout")))
    # aggregate deltas
    d_active = [r["active_best_eff"] - r["pointwise_eff"] for r in rows]
    d_rr = [r["roundrobin_eff"] - r["pointwise_eff"] for r in rows]
    lines.append("SIMULATED pairwise (from pointwise scores); no pairwise verifier dumps exist.")
    lines.append(f"Selection efficiency P(pick correct | correct present), 3 families pooled-per-family:")
    for r in rows:
        lines.append(f"  {r['family']:<13} pointwise={r['pointwise_eff']:.3f}  "
                     f"round-robin={r['roundrobin_eff']:.3f}  knockout={r['knockout_eff']:.3f}  "
                     f"active-IDS(best)={r['active_best_eff']:.3f} @cost={r['active_best_cost']:.1f}")
    lines.append(f"active-IDS - pointwise (sel_eff): {['%+.3f'%x for x in d_active]}  "
                 f"(mean {np.mean(d_active):+.3f})")
    lines.append(f"round-robin(noise-free ceiling) - pointwise: {['%+.3f'%x for x in d_rr]}  "
                 f"(mean {np.mean(d_rr):+.3f})")
    beats = np.mean(d_active) > 0.003
    lines.append("")
    if beats:
        lines.append(f"=> C9 BEATS pointwise-argmax on selection by mean {np.mean(d_active):+.3f} (under simulation).")
    else:
        lines.append("=> C9 does NOT beat pointwise-argmax on selection accuracy under simulated pairwise")
        lines.append("   (round-robin ceiling ~= pointwise confirms the pairwise preference carries no")
        lines.append("   information beyond the pointwise score). C9's value is COST vs a FULL pairwise pass:")
    # cost story from held-out
    for r in rows:
        h = r["heldout"]
        if h:
            rr_cost = 28.0
            lines.append(f"   {r['family']:<13} active-IDS matches round-robin selection with "
                         f"~{h['cost']:.1f}/{rr_cost:.0f} comparisons ({(1-h['cost']/rr_cost)*100:.0f}% fewer "
                         f"than full pairwise; pointwise uses 0).")
    lines.append("")
    lines.append("HONEST LIMIT: to test whether active comparison beats pointwise on ACCURACY needs REAL")
    lines.append("pairwise verifier verdicts. GPU experiment sized in RESULT['gpu_experiment'].")
    # size the GPU experiment
    total_q = sum(r["n"] for r in rows)
    RESULT["gpu_experiment"] = dict(
        why="simulated pairwise cannot exceed pointwise; need a real (img,q,ansA,ansB)->A/B comparator",
        full_pairwise_calls_per_q="N(N-1)/2 = 28 for N=8",
        active_ids_calls_per_q_est=float(np.mean([r["heldout"]["cost"] for r in rows if r["heldout"]])) if any(r["heldout"] for r in rows) else None,
        total_questions=int(total_q),
        note=("Train the B1 pairwise head (reuse the ~6k verifier pairs re-formed as within-question A/B), "
              "then dump only the ~O(N) comparisons active-IDS actually schedules (not all 28) -> a fraction "
              "of a full pairwise pass. Re-run THIS script pointing at the real pairwise verdicts to get the "
              "true accuracy test."),
    )
    return dict(lines=lines, rows=rows, mean_active_minus_pointwise=float(np.mean(d_active)),
                mean_roundrobin_minus_pointwise=float(np.mean(d_rr)), beats_pointwise=bool(beats))


# ================================================================ EXPERIMENT 2: REAL-verdict driver
MCQ_DUMP_REAL = {
    "vqa_rad_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_VQA_RAD_lingshu7b_VQA_RAD_content_content_sc8.jsonl",
    "slake_open":   "ckpts/mcq_gen_verify/lingshu7b/ckpt_SLAKE_lingshu7b_SLAKE_content_content_sc8.jsonl",
    "pathvqa_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_PATH_VQA_lingshu7b_content_sc8.jsonl",
}


def load_real_verdicts(verdict_dir, tag):
    """(ds,idx)->{(i,j): p_i_gt_j}. Also returns set of (ds,idx) with >=1 verdict (coverage)."""
    global _REAL
    _REAL = {}
    covered = set()
    for ds in MCQ_DUMP_REAL:
        vp = J(os.path.join(verdict_dir, f"verdicts_{ds}_{tag}.jsonl"))
        if not os.path.exists(vp):
            continue
        for l in open(vp):
            if not l.strip():
                continue
            r = json.loads(l)
            key = (ds, str(r["idx"]))
            _REAL.setdefault(key, {})[(int(r["i"]), int(r["j"]))] = float(r["p_i_gt_j"])
            covered.add(key)
    return covered


def load_family_real(covered):
    """Build per-dataset rows from the iid mcq dumps, restricted to questions that have REAL verdicts."""
    per_ds = {}
    for ds, dump in MCQ_DUMP_REAL.items():
        dp = J(dump)
        if not os.path.exists(dp):
            continue
        rows = []
        for l in open(dp):
            if not l.strip():
                continue
            r = json.loads(l)
            idx = str(r["idx"])
            if (ds, idx) not in covered:
                continue
            sl = [0 if x in (None, -1) else int(x) for x in r["sl"]]
            sc = list(r["scores"])
            n = min(len(sl), len(sc))
            if n < 2:
                continue
            rows.append(dict(sl=sl[:n], sc=sc[:n], greedy_ok=int(r.get("greedy_ok", sl[0])),
                             ds=ds, idx=idx))
        if rows:
            per_ds[ds] = rows
    return per_ds


def real_main(verdict_dir, tag):
    covered = load_real_verdicts(verdict_dir, tag)
    per_ds = load_family_real(covered)
    print("=" * 108)
    print("EXPERIMENT 2  REAL pairwise verifier verdicts (Lingshu-7B pooled4 LoRA in pairwise mode)")
    print("    pairwise outcomes are ACTUAL (img,q,ansA,ansB)->A/B verdicts, position-debiased -- NOT simulated")
    print("=" * 108)
    if not per_ds:
        print("[real] no rows with verdicts found under", verdict_dir); return
    fam = run_family("Lingshu(real-pairwise)", per_ds)
    RESULT = dict(
        mode="REAL_PAIRWISE",
        note=("Pairwise verdicts are REAL: same weights as the pointwise verifier (Lingshu-7B + pooled4 LoRA) "
              "prompted A-vs-B, both orders averaged for position debias. This tests whether a pairwise/active-"
              "comparison verifier beats the POINTWISE verifier on selection accuracy (the 74-82% ceiling)."),
        verdict_dir=verdict_dir, tag=tag,
        params=dict(BETA=BETA, SIGMA0=SIGMA0, BETA_TS=BETA_TS, MC=MC, NEARTIE=NEARTIE, SEEDS=SEEDS,
                    PCONF_GRID=PCONF_GRID),
        families={"Lingshu(real-pairwise)": fam},
    )
    p = fam["pooled"]
    print(f"\n############  REAL pairwise (n_datasets={fam['n_datasets']}, n={p['pointwise']['n']})  ############")
    print(f"  SC/greedy      {fmt(p['greedy'])}")
    print(f"  oracle@N       {fmt(p['oracle@N'])}   <- ceiling")
    print(f"  POINTWISE (dep){fmt(p['pointwise'])}   <- current method, 0 comparisons")
    print(f"  knockout(real) {fmt(p['knockout'])}")
    print(f"  round-robin(real Copeland) {fmt(p['copeland_det'])}  <- REAL pairwise, all 28 pairs")
    print(f"  active-IDS(real) frontier (p_conf -> acc/eff/mean_cmp):")
    for f in p["active_frontier"]:
        print(f"      p_conf={f['p_conf']:.2f}   acc={f['sel_acc']:.3f} eff={f['sel_eff']:.3f} mean_cmp={f['cost']:.2f}")
    if "neartie" in p:
        nt = p["neartie"]
        print(f"  NEAR-TIE subset (top-2 pointwise gap<{NEARTIE}, correct present; n={nt['n']}) selection acc:")
        print(f"      pointwise={nt['pointwise']:.3f}  knockout(real)={nt['knockout']:.3f}  "
              f"round-robin(real)={nt['copeland_det']:.3f}  active(real)@p90={nt['active_p90']:.3f}")

    # verdict: does REAL pairwise beat pointwise?
    pw = p["pointwise"]; rr = p["copeland_det"]; ko = p["knockout"]
    af_eff = max(p["active_frontier"], key=lambda f: (f["sel_eff"], -f["cost"]))
    af_acc = max(p["active_frontier"], key=lambda f: (f["sel_acc"], -f["cost"]))
    d_rr_eff = rr["sel_eff"] - pw["sel_eff"]; d_rr_acc = rr["sel_acc"] - pw["sel_acc"]
    lines = [
        "REAL pairwise verdicts (NOT simulated).",
        f"pooled n={pw['n']} over {fam['n_datasets']} datasets.",
        f"selection ACCURACY  pointwise={pw['sel_acc']:.3f}  round-robin(real)={rr['sel_acc']:.3f} "
        f"(Δ{d_rr_acc:+.3f})  knockout(real)={ko['sel_acc']:.3f}  active(real,best)={af_acc['sel_acc']:.3f}",
        f"selection EFFICIENCY pointwise={pw['sel_eff']:.3f}  round-robin(real)={rr['sel_eff']:.3f} "
        f"(Δ{d_rr_eff:+.3f})  knockout(real)={ko['sel_eff']:.3f}  active(real,best)={af_eff['sel_eff']:.3f}",
    ]
    beats = (d_rr_acc > 0.003) or (af_acc["sel_acc"] - pw["sel_acc"] > 0.003)
    if beats:
        lines.append(f"=> REAL pairwise BEATS pointwise-argmax on selection accuracy (Δacc up to "
                     f"{max(d_rr_acc, af_acc['sel_acc']-pw['sel_acc']):+.3f}).")
    else:
        lines.append("=> REAL pairwise does NOT beat pointwise-argmax on selection accuracy "
                     f"(round-robin Δacc={d_rr_acc:+.3f}, best active Δacc={af_acc['sel_acc']-pw['sel_acc']:+.3f}).")
    RESULT["verdict"] = dict(lines=lines, pointwise_sel_acc=pw["sel_acc"], pointwise_sel_eff=pw["sel_eff"],
                             roundrobin_real_sel_acc=rr["sel_acc"], roundrobin_real_sel_eff=rr["sel_eff"],
                             knockout_real_sel_acc=ko["sel_acc"], active_real_best_sel_acc=af_acc["sel_acc"],
                             beats_pointwise=bool(beats), n=pw["n"], n_datasets=fam["n_datasets"])
    RESULT["per_dataset_pointwise_vs_real"] = {
        ds: {"n": fam["datasets"][ds]["pointwise"]["n"],
             "pointwise_sel_acc": fam["datasets"][ds]["pointwise"]["sel_acc"],
             "pointwise_sel_eff": fam["datasets"][ds]["pointwise"]["sel_eff"],
             "roundrobin_real_sel_acc": fam["datasets"][ds]["copeland_det"]["sel_acc"],
             "roundrobin_real_sel_eff": fam["datasets"][ds]["copeland_det"]["sel_eff"],
             "knockout_real_sel_acc": fam["datasets"][ds]["knockout"]["sel_acc"],
             "oracle_sel_acc": fam["datasets"][ds]["oracle@N"]["sel_acc"]}
        for ds in fam["datasets"]
    }
    print("\n" + "=" * 108 + "\nVERDICT (Experiment 2)\n" + "=" * 108)
    for l in lines: print("  " + l)
    outp = J("results/cascade_methods/artifacts/pairwise_verifier_gpu.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(RESULT, open(outp, "w"), indent=1)
    print(f"\n[dump] {outp}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_verdicts_dir", default=None,
                    help="dir with verdicts_<ds>_<tag>.jsonl (Experiment 2). If set, runs the REAL-pairwise "
                         "analysis and writes pairwise_verifier_gpu.json. If unset, runs the SIMULATED analysis.")
    ap.add_argument("--tag", default="lingshu7b")
    A = ap.parse_args()
    if A.real_verdicts_dir:
        real_main(A.real_verdicts_dir, A.tag)
    else:
        main()
