#!/usr/bin/env python3
"""
escalation_more.py -- OFFLINE test of the top UNTESTED backlog sec-G escalation levers (G7 / G2 / G4)
applied to the integrated format-aware cascade (integrated_method.py).  Companion to escalation_levers.py
(which tested G5 / G6 / G8; G3 was tested in quantized_strong_leg.py).  NO new inference: everything is
re-simulated from saved per-sample dumps + the repo's measured batch-1 cost constants.

The three UNTESTED offline-adjacent ideas, per METHOD_IDEAS_BACKLOG.md sec-G:

  G7  SEMANTIC ESCALATION CACHE (rarer, amortized).  Key 32B answers by (image, question); a near-duplicate
      escalation hits the cache -> the 32B fires once per cluster over a stream.  DATA-LIMITED: no dump carries
      an image id/hash, so the only available key is normalized question TEXT.  We therefore (a) measure the
      question-text near-duplicate rate + within-cluster gold agreement per benchmark, and (b) simulate a
      question-keyed stream cache on the escalated set -> effective escalation-rate reduction AND its accuracy
      cost from unsafe cross-image reuse.  The correct image-keyed / embedding cache is flagged as future work.

  G2  EARLY-EXIT / PATIENCE HALTING (cheaper).  Task framing: is there a cheap early signal that lets a leg
      commit WITHOUT full generation on easy items?  OFFLINE-CHECK of the generated-token axis via gen_toks;
      the real layer-depth early-exit (CALM/LayerSkip) needs intermediate-layer logits -> GPU (flagged).

  G4  IMAGE-TOKEN PRUNING OF THE 32B PREFILL (cheaper).  Analytical re-cost: prefill is ~59% of the 665 ms
      strong forward (phi=0.586, measured) and image tokens are 79-94% of the prompt (token_cache.json), so
      pruning p% of image tokens after a shallow layer cuts prefill ~linearly.  Project per-escalation latency/
      FLOPs saving; ACCURACY IMPACT NEEDS A GPU CONFIRM (marked, not claimed).

COST CONSTANTS (batch-1, measured; identical to integrated_method / escalation_levers):
  GEN7 347 ms / 1.0 ; VER7 175 ms ; BO8 522 ms / 16.0 ; GEN32-nothink 665 ms / 4.57.
  32B prefill fraction phi = 0.586 (prefill 390 ms, decode 275 ms) -- measured, latency_32b.jsonl.

CPU only.  Launch from repo root:  python3 src/cascade_methods/escalation_more.py
"""
import json, os, re, sys
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import integrated_method as IM

ROOT = IM.ROOT
MEK = IM.MEK
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/escalation_more.json")

GEN7, VER7, BO8, GEN32N = IM.GEN7, IM.VER7, IM.BO8, IM.GEN32N
ALWAYS32NT = GEN32N["ms"]                     # 665 ms
PHI = 0.586                                   # measured 32B prefill fraction (escalation_levers header)
PREFILL32_MS = PHI * GEN32N["ms"]             # 390 ms
DECODE32_MS = (1 - PHI) * GEN32N["ms"]        # 275 ms
PREFILL32_FLOP = PHI * GEN32N["flop"]         # 2.68 FLOP-eq
# FastV/QG-VTC style: prune image tokens AFTER a shallow prefix of layers.  32B ~= 64 layers; drop at layer ~3
# -> the first K=3 layers still see all tokens, layers K+1..L see (1-p) of the image tokens.  Retained-saving
# factor on the prefill = (1 - K/L).  K/L=3/64=0.047 -> 0.953.  Conservative (attention's quadratic term saves
# even more; we model only the dominant LINEAR-in-token prefill term).
LAYER_RETAIN = 1.0 - 3.0 / 64.0

# ---------- normalized text ----------
def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", str(s).strip().lower())).strip()


# ============================ raw slice loader (ok/margin + question/gold aligned) =================
def raw_mcq(ds, closed=None, qkey="question"):
    """Mirror IM.mcq_closed's filter exactly, but ALSO return aligned normalized question + gold, and the
       stream order (raw idx)."""
    r7 = IM.load_raw("lingshu7b_full", ds); r32 = IM.load_raw("lingshu32b_full", ds)
    if r7 is None or r32 is None: return None
    n = min(len(r7), len(r32))
    if closed == "SLAKE":
        idx = [i for i in range(n) if r7[i].get("answer_type") == "CLOSED"]
    elif closed == "YESNO":
        idx = [i for i in range(n) if str(r7[i].get("answer", "")).strip().lower() in ("yes", "no")]
    else:
        idx = list(range(n))
    ok7 = np.array([IM.as_ok(r7[i]) for i in idx], float)
    ok32 = np.array([IM.as_ok(r32[i]) for i in idx], float)
    mar = np.array([IM.as_float(r7[i].get("margin")) for i in idx], float)
    q = [norm(r7[i].get(qkey, "")) for i in idx]
    gold = [norm(r7[i].get("answer", "")) for i in idx]
    return dict(ok7=ok7, ok32=ok32, gate=mar, q=q, gold=gold, order=list(idx), cheap=GEN7)


def raw_slake_open():
    """SLAKE-open: EXACTLY IM.open_slake's ok7 (=LLM-judge verdict on the greedy answer), gate=seqlogprob,
       strong 32B judge (ok32); plus question/gold from the raw cheap jsonl for the cache key."""
    cj = IM.load_judge_jsonl(f"{ROOT}/ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b.judge.jsonl")
    sj = IM.load_judge_jsonl(f"{ROOT}/ckpts/openvqa/strong_lingshu/ckpt_slake_open_lingshu32b.judge.jsonl")
    seq, qg = {}, {}
    p = f"{ROOT}/ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b.jsonl"
    for l in open(p):
        if l.strip():
            r = json.loads(l); i = int(r["idx"])
            seq[i] = IM.as_float(r.get("seqlogprob")); qg[i] = (norm(r["question"]), norm(r["gold"]))
    ids = sorted(set(cj) & set(sj) & set(seq))          # same intersection as IM.open_slake
    ok7 = np.array([cj[i] for i in ids], float)
    ok32 = np.array([sj[i] for i in ids], float)
    gate = np.array([seq[i] for i in ids], float)
    q = [qg[i][0] for i in ids]; gold = [qg[i][1] for i in ids]
    return dict(ok7=ok7, ok32=ok32, gate=gate, q=q, gold=gold, order=ids, cheap=GEN7)


def load_slices():
    S = []
    for name, ds, closed, qk in [("PMC_VQA", "PMC_VQA", None, "prompt"),
                                 ("SLAKE_closed", "SLAKE", "SLAKE", "question"),
                                 ("VQA_RAD_closed", "VQA_RAD", "YESNO", "question"),
                                 ("PATH_VQA_closed", "PATH_VQA", "YESNO", "question"),
                                 ("MedXpertQA-MM", "MedXpertQA-MM", None, "question")]:
        d = raw_mcq(ds, closed, qk)
        if d: d["name"] = name; d["fmt"] = "MCQ"; d["imgds"] = {"PMC_VQA": "PMC-VQA", "SLAKE_closed": "SLAKE",
            "VQA_RAD_closed": "VQA-RAD", "PATH_VQA_closed": "PathVQA", "MedXpertQA-MM": "MedXpert-Reasoning"}[name]; S.append(d)
    d = raw_slake_open()
    d["name"] = "SLAKE_open"; d["fmt"] = "open"; d["imgds"] = "SLAKE"; S.append(d)
    return S


# ============================ held-out escalation mask (same cross-fit as IM) ======================
def heldout_esc(ok7, ok32, gate, K=5):
    n = len(ok7); esc = np.zeros(n, bool)
    for f in range(K):
        te = np.array([i % K == f for i in range(n)]); tr = ~te
        if tr.sum() < 2 or te.sum() < 1: continue
        tau = IM.pick_tau_isocost(ok7[tr], ok32[tr], gate[tr], ok32[tr].mean())
        esc[te] = gate[te] < tau
    return esc


# ============================ G7: semantic (question-keyed) escalation cache ========================
def g7_dupstats(S):
    """Per-benchmark question-text near-duplicate rate + within-cluster gold agreement (the safety bound)."""
    out = {}
    for s in S:
        q, gold = s["q"], s["gold"]; n = len(q)
        c = Counter(q); distinct = len(c)
        clusters = defaultdict(list)
        for i, qq in enumerate(q): clusters[qq].append(i)
        multi = [cl for cl in clusters.values() if len(cl) > 1]
        rows_multi = sum(len(cl) for cl in multi)
        same, tot = 0, 0
        for cl in multi:
            gs = [gold[i] for i in cl]; mode = Counter(gs).most_common(1)[0][0]
            same += sum(1 for g in gs if g == mode); tot += len(cl)
        out[s["name"]] = dict(n=n, distinct_questions=distinct, dup_rate=round(1 - distinct / n, 4),
                              rows_in_multi_question=rows_multi, frac_rows_in_multi=round(rows_multi / n, 4),
                              within_cluster_gold_agreement=round(same / tot, 4) if tot else None)
    return out


def g7_stream_cache(s, esc):
    """Simulate a question-keyed cache over the slice stream (order = raw idx).  When escalation fires and the
       question was seen+escalated before, REUSE the representative's 32B answer (no 32B call).  Reuse is
       correct iff member.gold == representative.gold AND representative's 32B was correct (=> its answer == the
       shared gold => correct for the member too).  Honest bound: no image key -> cross-image reuse is scored
       as WRONG where golds differ.  Returns effective escalation rate + accuracy of the cached cascade."""
    order = np.argsort(s["order"])           # stream order by raw idx
    q, gold, ok7, ok32 = s["q"], s["gold"], s["ok7"], s["ok32"]
    cache = {}                               # qnorm -> dict(gold, ok32)  (representative = first escalated)
    calls = 0; correct = 0
    for j in order:
        if not esc[j]:
            correct += int(ok7[j] == 1); continue
        key = q[j]
        if key in cache:                     # CACHE HIT: reuse, no 32B call
            rep = cache[key]
            correct += int(rep["gold"] == gold[j] and rep["ok32"] == 1)
        else:                                # MISS: pay the 32B, store it
            calls += 1
            cache[key] = dict(gold=gold[j], ok32=int(ok32[j]))
            correct += int(ok32[j] == 1)
    n = len(order)
    eff_esc = calls / n
    acc = correct / n
    return eff_esc, acc


# ============================ G4: image-token pruning of the 32B prefill (analytical) ===============
def img_frac(imgds, res="fullres"):
    """Mean image-token FRACTION of the prompt for this dataset at a resolution (token_cache.json)."""
    tc = json.load(open(f"{ROOT}/ckpts/token_cache.json"))
    if imgds not in tc: return None
    rows = list(tc[imgds][res].values())
    img = np.mean([v[1] for v in rows]); tot = np.mean([v[0] for v in rows])
    return float(img / tot)


def g4_esc_cost(imf, p):
    """Per-escalation 32B latency/FLOPs after pruning fraction p of image tokens (FastV shallow-exit model,
       LINEAR-in-token prefill).  saved_prefill_fraction = LAYER_RETAIN * p * imf."""
    saved_frac = LAYER_RETAIN * p * imf
    new_prefill_ms = PREFILL32_MS * (1 - saved_frac)
    new_ms = new_prefill_ms + DECODE32_MS
    new_prefill_flop = PREFILL32_FLOP * (1 - saved_frac)
    new_flop = new_prefill_flop + (GEN32N["flop"] - PREFILL32_FLOP)
    return new_ms, new_flop, saved_frac


# ============================ main ================================================================
def run():
    S = load_slices()
    names = [s["name"] for s in S]
    N = {s["name"]: len(s["ok7"]) for s in S}

    # ---- baseline held-out cascade + escalation profile ----
    base = {}; esc_masks = {}
    for s in S:
        esc = heldout_esc(s["ok7"], s["ok32"], s["gate"])
        esc_masks[s["name"]] = esc
        acc = float(np.where(esc, s["ok32"], s["ok7"]).mean()); e = float(esc.mean())
        base[s["name"]] = dict(acc=round(acc, 4), esc=round(e, 4),
                               latency_ms=round(s["cheap"]["ms"] + e * GEN32N["ms"], 1),
                               flops=round(s["cheap"]["flop"] + e * GEN32N["flop"], 3),
                               cheap_ms=s["cheap"]["ms"])
    # escalation-heavy cells (task focus)
    heavy = sorted([n for n in names], key=lambda n: -base[n]["esc"])

    # ======================== G7 ========================
    dup = g7_dupstats(S)
    g7 = {}
    for s in S:
        e = esc_masks[s["name"]]
        eff_esc, cache_acc = g7_stream_cache(s, e)
        b = base[s["name"]]
        saved_calls_frac = (b["esc"] - eff_esc)
        g7[s["name"]] = dict(
            base_esc=b["esc"], base_acc=b["acc"],
            cache_eff_esc=round(eff_esc, 4),
            esc_reduction_abs=round(saved_calls_frac, 4),
            esc_reduction_pct=round(100 * saved_calls_frac / b["esc"], 1) if b["esc"] > 0 else 0.0,
            cache_acc=round(cache_acc, 4),
            acc_cost_vs_base=round(cache_acc - b["acc"], 4),
            new_latency_ms=round(s["cheap"]["ms"] + eff_esc * GEN32N["ms"], 1),
            latency_saved_ms=round((b["esc"] - eff_esc) * GEN32N["ms"], 1),
            dup_rate=dup[s["name"]]["dup_rate"],
            within_cluster_gold_agreement=dup[s["name"]]["within_cluster_gold_agreement"])

    # ======================== G2 (gen_toks / early-exit data check) ========================
    gen_stats = {}
    for tag, lab in [("lingshu7b_full", "7B_cheap"), ("lingshu32b_full", "32B_strong")]:
        for ds in ["VQA_RAD", "MedXpertQA-MM", "SLAKE", "PMC_VQA", "PATH_VQA"]:
            r = IM.load_raw(tag, ds)
            if r is None: continue
            gt = np.array([(x.get("gen_toks") or 0) for x in r], float)
            gen_stats.setdefault(lab, {})[ds] = dict(mean=round(float(gt.mean()), 2),
                                                      median=int(np.median(gt)), p90=int(np.percentile(gt, 90)),
                                                      max=int(gt.max()))
    # theoretical MAX generated-token halt saving on a 32B escalation: decode is 275 ms over ~median tokens;
    # you cannot emit fewer than 1 answer token, and the answer *is* ~3 tokens, so realizable halt ~= 0.
    med_dec = 3
    g2_max_halt_ms = round(DECODE32_MS * (1 - 1.0 / med_dec), 1)   # ceiling if only 1 of 3 tokens were needed

    # ======================== G4 (image-token prune re-cost) ========================
    prune_grid = [0.25, 0.50, 0.75]
    g4 = {}
    for s in S:
        imf = img_frac(s["imgds"], "fullres")
        imf_cap = img_frac(s["imgds"], "cap320")
        b = base[s["name"]]; e = b["esc"]
        per_p = {}
        for p in prune_grid:
            new_ms, new_flop, sf = g4_esc_cost(imf, p)
            slice_lat = s["cheap"]["ms"] + e * new_ms
            slice_flop = s["cheap"]["flop"] + e * new_flop
            per_p["p=%.2f" % p] = dict(
                esc_ms=round(new_ms, 1), esc_ms_saved=round(GEN32N["ms"] - new_ms, 1),
                esc_flop=round(new_flop, 3), prefill_saved_frac=round(sf, 3),
                slice_latency_ms=round(slice_lat, 1),
                slice_latency_saved_vs_base_ms=round(b["latency_ms"] - slice_lat, 1),
                slice_flops=round(slice_flop, 3))
        g4[s["name"]] = dict(fullres_img_frac=round(imf, 3), cap320_img_frac=round(imf_cap, 3),
                             base_esc=e, base_latency_ms=b["latency_ms"], per_prune=per_p)

    # pooled G4 at p=0.50 over the scored slices (sample-weighted)
    tot = sum(N.values())
    def poolstat(getter): return sum(getter(n) * N[n] for n in names) / tot
    g4_pool = {}
    for p in prune_grid:
        lat = poolstat(lambda n: g4[n]["per_prune"]["p=%.2f" % p]["slice_latency_ms"])
        flp = poolstat(lambda n: g4[n]["per_prune"]["p=%.2f" % p]["slice_flops"])
        g4_pool["p=%.2f" % p] = dict(latency_ms=round(lat, 1), flops=round(flp, 3))
    base_pool_lat = poolstat(lambda n: base[n]["latency_ms"])
    base_pool_flop = poolstat(lambda n: base[n]["flops"])

    # ======================== assemble ========================
    out = dict(
        what="OFFLINE test of UNTESTED backlog sec-G escalation levers G7 (semantic cache) / G2 (early-exit) / "
             "G4 (32B image-token prune) on the integrated cascade; re-simulated on saved dumps + measured "
             "batch-1 costs.  Companion to escalation_levers.json (G5/G6/G8) and quantized_strong_leg.json (G3).",
        cost_constants=dict(GEN7=GEN7, VER7=VER7, BO8=BO8, GEN32_nothink=GEN32N,
                            phi_prefill_fraction=PHI, prefill32_ms=round(PREFILL32_MS, 1),
                            decode32_ms=round(DECODE32_MS, 1),
                            g4_layer_retain="1 - 3/64 = %.3f (FastV shallow-exit at layer 3 of ~64)" % LAYER_RETAIN),
        baseline_escalation_profile={n: base[n] for n in heavy},
        escalation_heavy_cells=[n for n in heavy if base[n]["esc"] >= 0.40],

        G7_semantic_escalation_cache=dict(
            idea="fire the 32B once per near-duplicate (image, question) cluster over a stream; a cache hit skips "
                 "the 32B.  DATA-LIMITED: no dump carries an image id/hash -> only normalized question TEXT is a "
                 "usable key, which conflates different images that share a templated question.",
            per_benchmark_dup=dup,
            question_keyed_cache_sim=g7,
            verdict=("G7 gives ~ZERO relief where it is needed and is UNSAFE where duplication exists.  On the "
                     "escalation-heavy cells the question-text near-duplicate rate is ~0 (MedXpert 0.000, PMC 0.001, "
                     "VQA_RAD_closed 0.064) -> ~no cache hits, ~no saving.  The only high-duplication slices "
                     "(SLAKE_closed 0.815, SLAKE_open 0.738, PATH_VQA_closed 0.371) are templated questions asked "
                     "over DIFFERENT images: within a question cluster the gold answer agrees only 0.57 (SLAKE_open) "
                     "/ 0.89 (SLAKE_closed) / 0.91 (PathVQA_closed), so a question-keyed cache returns a WRONG "
                     "cross-image answer on the disagreeing fraction -- e.g. SLAKE_open cuts effective escalation "
                     "58%% but at -0.119 accuracy; SLAKE_closed's spurious +0.036 is a templated-reuse artifact, not "
                     "a real gain.  The correct key is (image-hash + question-embedding); NO image id exists in any "
                     "dump, so the safe/semantic cache is UNMEASURABLE offline here and is future work.  Benchmark "
                     "duplication is a templated-text artifact; a real deployment stream's (image,question) repeat "
                     "rate is a distribution assumption, not present in this eval."),
            data_gap="No image id/hash/path in ANY per-sample dump -> the cache's true key is not reconstructable "
                     "offline; question-text-only clustering OVERCOUNTS duplicates (different images, same template)."),

        G2_early_exit_patience=dict(
            idea="commit a leg without full generation on easy items (generated-token patience OR layer-depth "
                 "early-exit).  Task asks: is the signal in the dumps?",
            gen_toks_distribution=gen_stats,
            finding=("BOTH the 7B cheap leg and the 32B strong leg emit ~3 generated tokens on EVERY benchmark "
                     "(median 3, p90 3-7): the answer is a letter/word, so both legs are PREFILL-BOUND.  There is "
                     "essentially no generation to halt -- generated-token patience saves ~0.  Theoretical ceiling "
                     "if only 1 of ~3 answer tokens were needed = %.0f ms/escalation, but the answer IS ~3 tokens, "
                     "so the realizable generated-token saving is ~0." % g2_max_halt_ms),
            real_lever_needs_gpu=("The lever with real headroom is LAYER-DEPTH early-exit (CALM/LayerSkip): exit the "
                                  "32B forward at an intermediate layer once per-token confidence is high (literature "
                                  "~2-3x on the WHOLE forward incl. prefill).  That needs intermediate-layer logits / "
                                  "an exit head, which are NOT in the dumps -> GPU probe required.  A cheap offline "
                                  "confirm path: dump the 32B's per-layer top-1 agreement with its final answer on a "
                                  "small sample to bound the exit layer."),
            verdict="G2 is NOT offline-testable for its real mechanism (layer-exit -> GPU).  The offline-visible "
                    "generated-token axis is already saturated (~3 tokens, prefill-bound) so it offers no saving.  "
                    "DATA-ABSENT for the productive variant; flagged GPU."),

        G4_image_token_prune_32B_prefill=dict(
            idea="prune p% of the 32B's image tokens after a shallow layer (FastV/QG-VTC); prefill is ~59%% of the "
                 "665 ms forward and image tokens are 79-94%% of the prompt, so the per-escalation cost drops "
                 "~linearly with pruned image tokens.  ACCURACY IMPACT NEEDS A GPU CONFIRM (projection only).",
            model=("per-escalation 32B latency = prefill*(1 - LAYER_RETAIN*p*img_frac) + decode ; "
                   "prefill=%.0f ms, decode=%.0f ms, LAYER_RETAIN=%.3f.  LINEAR-in-token prefill (dominant term); "
                   "the quadratic attention term saves even more, so this is conservative." %
                   (PREFILL32_MS, DECODE32_MS, LAYER_RETAIN)),
            per_slice=g4,
            pooled=dict(pool_scope="sample-weighted over the 6 loaded slices; DOMINATED by PMC (33k rows, 8.5%% "
                        "esc), so the pooled % understates the win -- the per-escalation and heavy-cell numbers are "
                        "the headline.",
                        base_latency_ms=round(base_pool_lat, 1), base_flops=round(base_pool_flop, 3),
                        pruned=g4_pool,
                        latency_saved_pct={p: round(100 * (1 - g4_pool[p]["latency_ms"] / base_pool_lat), 1) for p in g4_pool},
                        flops_saved_pct={p: round(100 * (1 - g4_pool[p]["flops"] / base_pool_flop), 1) for p in g4_pool}),
            verdict=("G4 is the best NEW offline-projected speed lever of the three.  At p=0.50 image-token prune "
                     "the 32B escalation forward drops ~665 -> ~590-620 ms (per-slice, PROJECTED) with the largest "
                     "absolute win on the escalation-heavy, image-token-rich cells (VQA_RAD_closed img_frac 0.94, "
                     "MedXpert 0.79, SLAKE_open 0.91).  This is a per-escalation (cheaper) lever, so it stacks with "
                     "the escalation-RATE levers.  It is a PROJECTION: the FLOPs/latency model is analytical (real, "
                     "from measured phi + token_cache), but the ACCURACY at each prune level MUST be confirmed on a "
                     "GPU (medical fine detail may sit in low-salience tokens -> per-benchmark guardrail).")),

        MEASURED_VS_PROJECTED=dict(
            G7="MEASURED (dup rates, gold agreement, question-keyed cache esc-reduction + acc-cost are computed on "
               "real dumps).  The SAFE/semantic image-keyed version is UNMEASURABLE (no image id) -> future work.",
            G2="MEASURED that answers are ~3 tokens (prefill-bound) -> generated-token halt ~0.  Layer-exit lever = "
               "PROJECTED/literature, needs GPU.",
            G4="PROJECTED latency/FLOPs (analytical model on measured phi=0.586 + real token_cache img_frac).  "
               "Accuracy = NEEDS GPU CONFIRM."),

        VERDICT=dict(
            best_new_speed_lever="G4 (image-token pruning of the 32B prefill) -- the only one of the three with real, "
                                 "quantified offline headroom on the escalation-heavy cells.",
            g4_saving="PROJECTED per-escalation 32B latency/FLOPs -%.0f%% @ p=0.50 / -%.0f%% @ p=0.75 on the most "
                      "image-token-rich cell (VQA_RAD_closed); pooled cascade latency %.0f -> %.0f ms @ p=0.50 "
                      "(-%.1f%%).  ACCURACY NEEDS GPU CONFIRM." % (
                          100 * (GEN32N["ms"] - g4["VQA_RAD_closed"]["per_prune"]["p=0.50"]["esc_ms"]) / GEN32N["ms"],
                          100 * (GEN32N["ms"] - g4["VQA_RAD_closed"]["per_prune"]["p=0.75"]["esc_ms"]) / GEN32N["ms"],
                          base_pool_lat, g4_pool["p=0.50"]["latency_ms"],
                          100 * (1 - g4_pool["p=0.50"]["latency_ms"] / base_pool_lat)),
            g7="DEAD offline: ~0 duplication on the escalation-heavy cells (MedXpert/PMC/VQA_RAD), and unsafe "
               "cross-image reuse where duplication exists (SLAKE/PathVQA templated).  No image key in the dumps.",
            g2="DATA-ABSENT offline: answers are ~3 tokens (prefill-bound), so generated-token halting saves ~0; the "
               "real layer-depth early-exit needs a GPU probe.",
            bottom_line="G4 is the single new lever worth a GPU accuracy confirm; it is CHEAPER-per-escalation and "
                        "STACKS with the previously-validated G8 (prefill prefetch) and G5 (futility suppressor).  "
                        "G7 and G2 are offline dead-ends on THIS eval (no image key / prefill-bound)."),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    _console(out, base, heavy, dup, g7, gen_stats, g2_max_halt_ms, g4, g4_pool, base_pool_lat, base_pool_flop)
    print(f"\nwrote {OUT}")
    return out


def _console(out, base, heavy, dup, g7, gen_stats, g2_max_halt_ms, g4, g4_pool, base_pool_lat, base_pool_flop):
    B = "=" * 112
    print(B); print("UNTESTED ESCALATION LEVERS  G7 (semantic cache) / G2 (early-exit) / G4 (32B image-token prune)")
    print(B)
    print("\nBASELINE escalation profile (held-out, sorted by esc):")
    print(f"  {'slice':<18}{'esc%':>7}{'acc':>8}{'lat_ms':>9}{'flops':>8}")
    for n in heavy:
        b = base[n]
        print(f"  {n:<18}{b['esc']*100:>6.0f}%{b['acc']:>8.3f}{b['latency_ms']:>9.0f}{b['flops']:>8.3f}")

    print("\n--- G7 semantic escalation cache (question-keyed; NO image id in dumps) ---")
    print(f"  {'slice':<18}{'dupRate':>8}{'goldAgr':>8}{'baseEsc':>8}{'effEsc':>8}{'escRed%':>8}{'accCost':>9}")
    for n in g7:
        g = g7[n]; ga = g['within_cluster_gold_agreement']
        print(f"  {n:<18}{g['dup_rate']:>8.3f}{(ga if ga is not None else float('nan')):>8.3f}"
              f"{g['base_esc']:>8.3f}{g['cache_eff_esc']:>8.3f}{g['esc_reduction_pct']:>7.1f}%{g['acc_cost_vs_base']:>+9.4f}")
    print("  -> ~0 dup on escalation-heavy cells (MedXpert/PMC/VQA_RAD); SLAKE/PathVQA dup is templated over "
          "different images -> reuse is WRONG (see accCost).  DEAD offline.")

    print("\n--- G2 early-exit / patience (gen_toks) ---")
    print(f"  {'dataset':<16}{'7B med':>8}{'7B p90':>8}{'32B med':>9}{'32B p90':>9}")
    for ds in gen_stats["7B_cheap"]:
        a = gen_stats["7B_cheap"][ds]; c = gen_stats["32B_strong"][ds]
        print(f"  {ds:<16}{a['median']:>8}{a['p90']:>8}{c['median']:>9}{c['p90']:>9}")
    print(f"  -> answers ~3 tokens => PREFILL-BOUND; generated-token halt ceiling ~{g2_max_halt_ms:.0f} ms but "
          f"unrealizable (answer IS ~3 tok).  Real lever = layer-exit -> GPU.  DATA-ABSENT offline.")

    print("\n--- G4 image-token prune of the 32B prefill (PROJECTED latency/FLOPs; ACC needs GPU) ---")
    print(f"  {'slice':<18}{'imgFrac':>8}{'baseEsc':>8}   per-escalation 32B ms @ prune  (p=.25/.50/.75)")
    for n in g4:
        gg = g4[n]; pp = gg["per_prune"]
        print(f"  {n:<18}{gg['fullres_img_frac']:>8.3f}{gg['base_esc']:>8.3f}   "
              f"665 -> {pp['p=0.25']['esc_ms']:.0f} / {pp['p=0.50']['esc_ms']:.0f} / {pp['p=0.75']['esc_ms']:.0f}")
    print(f"  POOLED cascade latency: base {base_pool_lat:.0f} ms -> "
          f"{g4_pool['p=0.25']['latency_ms']:.0f}/{g4_pool['p=0.50']['latency_ms']:.0f}/{g4_pool['p=0.75']['latency_ms']:.0f} ms "
          f"(p=.25/.50/.75);  FLOPs base {base_pool_flop:.3f} -> "
          f"{g4_pool['p=0.25']['flops']:.3f}/{g4_pool['p=0.50']['flops']:.3f}/{g4_pool['p=0.75']['flops']:.3f}")

    print("\nVERDICT")
    for k, v in out["VERDICT"].items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    run()
