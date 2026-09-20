#!/usr/bin/env python3
"""Inventory + judge-coverage + dataset-alignment for every open-text generation dump under
MAIN/ckpts/openvqa (recursive). READ-ONLY. Writes out/inventory.json and out/inventory.md.

For every ckpt_<ds>_<tag>.jsonl (not *_scexploded*, not *.judge*, not *.audit*):
  n rows, unique idx, dup idx, bad json, preds/question min/median/max, mtime
  vs the dataset rebuilt by 01_build_reference.py: n_ref, missing idx, extra idx, question mismatches,
     gold mismatches (THE STALE-CHECKPOINT CLASS)
  exploded file: exists, rows, == recomputed explode of the dump (key set + per-key text)
  judge file (on exploded for pools, on the dump itself for greedy): rows, dup keys, coverage of expected keys,
     orphan keys, mtime ordering
"""
import json, os, re, time, statistics, sys
from collections import Counter, defaultdict
MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
R = f"{MAIN}/ckpts/openvqa"
OUT = "/data/dan/audit_2026-09-18/tmp/data-integrity/out"
ref = json.load(open(f"{OUT}/reference.json"))
REFMAP = {ds: {str(x["idx"]): x for x in v} for ds, v in ref.items()}
DS = sorted(ref, key=len, reverse=True)   # longest prefix first (pathvqa_open_train before pathvqa_open)

def parse(fn):
    b = fn[len("ckpt_"):-len(".jsonl")]
    for ds in DS:
        if b.startswith(ds + "_"):
            return ds, b[len(ds) + 1:]
    return None, b

def rd(p):
    rows, bad = [], 0
    for l in open(p):
        if not l.strip(): continue
        try: rows.append(json.loads(l))
        except Exception: bad += 1
    return rows, bad

def mt(p): return os.path.getmtime(p)
def ts(t): return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))

inv = []
for dp, dn, fn in os.walk(R):
    for f in sorted(fn):
        if not (f.startswith("ckpt_") and f.endswith(".jsonl")): continue
        if "_scexploded" in f or f.endswith(".judge.jsonl") or f.endswith(".audit.jsonl"): continue
        p = os.path.join(dp, f)
        ds, tag = parse(f)
        rows, bad = rd(p)
        rec = {"dir": os.path.relpath(dp, R), "file": f, "ds": ds, "tag": tag, "n_rows": len(rows), "bad_json": bad,
               "mtime": ts(mt(p)), "bytes": os.path.getsize(p)}
        if not rows or "idx" not in rows[0]:
            rec["note"] = "no idx / empty"; inv.append(rec); continue
        idxs = [str(r["idx"]) for r in rows]
        c = Counter(idxs)
        rec["n_unique_idx"] = len(c); rec["n_dup_idx_rows"] = len(idxs) - len(c)
        if "preds" in rows[0]:
            npred = [len(r.get("preds") or []) for r in rows]
            rec["preds_min"], rec["preds_med"], rec["preds_max"] = min(npred), statistics.median(npred), max(npred)
            rec["preds_hist"] = dict(Counter(npred).most_common(4))
            gt = [r.get("gen_tokens") for r in rows if isinstance(r.get("gen_tokens"), (int, float))]
            rec["gen_tokens_mean"] = round(sum(gt) / len(gt), 2) if gt else None
            rec["has_margin"] = "margin" in rows[0]; rec["has_lat"] = "lat_s" in rows[0]
            rec["keys_first"] = sorted(rows[0].keys()); rec["keys_last"] = sorted(rows[-1].keys())
            rec["schema_mixed"] = len({tuple(sorted(r.keys())) for r in rows}) > 1
        # ---- vs dataset
        if ds in REFMAP:
            RM = REFMAP[ds]
            rec["n_ref"] = len(RM)
            rec["missing_vs_ref"] = len(set(RM) - set(c)); rec["extra_vs_ref"] = len(set(c) - set(RM))
            qm = gm = 0
            for r in rows:
                x = RM.get(str(r["idx"]))
                if x is None: continue
                if "question" in r and str(r["question"]).strip() != str(x["question"]).strip(): qm += 1
                if "gold" in r and str(r["gold"]).strip() != str(x["gold"]).strip(): gm += 1
            rec["q_mismatch"], rec["gold_mismatch"] = qm, gm
            rec["n_compared_q"] = sum(1 for r in rows if str(r["idx"]) in RM and "question" in r)
        # ---- exploded + judge
        is_pool = "preds" in rows[0] and rec.get("preds_max", 1) > 1
        ep = p.replace(".jsonl", "_scexploded.jsonl")
        jp_direct = p.replace(".jsonl", ".judge.jsonl")
        rec["is_pool"] = is_pool
        if os.path.exists(ep):
            # recompute the explode
            exp_expected = {}
            firstrow = {}
            for r in rows:
                if str(r["idx"]) in firstrow: continue      # duplicate idx: exploder would emit both; note separately
                firstrow[str(r["idx"])] = r
                seen = {}
                for k, a in enumerate(r.get("preds") or []):
                    key = str(a).strip().lower()
                    if key in seen: continue
                    seen[key] = k
                    exp_expected[f'{r["idx"]}#{k}'] = (r.get("question"), r.get("gold"), a)
            erows, ebad = rd(ep)
            ekeys = [str(e["idx"]) for e in erows]
            ec = Counter(ekeys)
            rec["exp_rows"] = len(erows); rec["exp_dup_keys"] = len(ekeys) - len(ec); rec["exp_bad_json"] = ebad
            rec["exp_expected_rows"] = len(exp_expected)
            rec["exp_missing_keys"] = len(set(exp_expected) - set(ec)); rec["exp_extra_keys"] = len(set(ec) - set(exp_expected))
            rec["exp_missing_questions"] = len({k.split("#")[0] for k in set(exp_expected) - set(ec)})
            tm = 0
            for e in erows:
                x = exp_expected.get(str(e["idx"]))
                if x is None: continue
                if (e.get("question"), e.get("gold"), e.get("modal_pred")) != x: tm += 1
            rec["exp_text_mismatch"] = tm; rec["exp_n_text_compared"] = sum(1 for e in erows if str(e["idx"]) in exp_expected)
            rec["exp_mtime"] = ts(mt(ep)); rec["exp_older_than_dump"] = mt(ep) < mt(p)
            jp = ep.replace(".jsonl", ".judge.jsonl")
            if os.path.exists(jp):
                jrows, jbad = rd(jp)
                jk = [str(j["idx"]) for j in jrows]; jc = Counter(jk)
                rec["judge_rows"] = len(jrows); rec["judge_dup_keys"] = len(jk) - len(jc); rec["judge_bad_json"] = jbad
                rec["judge_cov_pairs"] = (len(set(jc) & set(exp_expected)) / len(exp_expected)) if exp_expected else None
                rec["judge_missing_pairs"] = len(set(exp_expected) - set(jc))
                rec["judge_orphans"] = len(set(jc) - set(exp_expected))
                qs_all = {k.split("#")[0] for k in exp_expected}
                qs_full = {q for q in qs_all}
                miss_q = {k.split("#")[0] for k in set(exp_expected) - set(jc)}
                rec["judge_questions_fully_covered"] = len(qs_all - miss_q); rec["judge_questions_total"] = len(qs_all)
                rec["judge_mtime"] = ts(mt(jp)); rec["judge_older_than_dump"] = mt(jp) < mt(p)
                rec["judge_older_than_exploded"] = mt(jp) < mt(ep)
                rec["judge_pos_rate"] = round(sum(int(j["judge_ok"]) for j in jrows) / max(1, len(jrows)), 4)
            else:
                rec["judge_rows"] = None
        elif is_pool:
            rec["exp_rows"] = None
        if os.path.exists(jp_direct):
            jrows, jbad = rd(jp_direct)
            jk = [str(j["idx"]) for j in jrows]; jc = Counter(jk)
            rec["djudge_rows"] = len(jrows); rec["djudge_dup"] = len(jk) - len(jc)
            rec["djudge_missing"] = len(set(c) - set(jc)); rec["djudge_orphans"] = len(set(jc) - set(c))
            rec["djudge_cov"] = len(set(jc) & set(c)) / len(c)
            rec["djudge_mtime"] = ts(mt(jp_direct)); rec["djudge_older_than_dump"] = mt(jp_direct) < mt(p)
            rec["djudge_pos_rate"] = round(sum(int(j["judge_ok"]) for j in jrows) / max(1, len(jrows)), 4)
        inv.append(rec)

# stray exploded / judge files with no parent dump
allf = set()
for dp, dn, fn in os.walk(R):
    for f in fn: allf.add(os.path.join(dp, f))
stray = []
for p in sorted(allf):
    if p.endswith("_scexploded.jsonl") and p.replace("_scexploded.jsonl", ".jsonl") not in allf: stray.append(p)
    if p.endswith("_scexploded.judge.jsonl") and p.replace(".judge.jsonl", ".jsonl") not in allf: stray.append(p)
    if p.endswith(".judge.jsonl") and "_scexploded" not in p and p.replace(".judge.jsonl", ".jsonl") not in allf: stray.append(p)
    if os.path.getsize(p) == 0: stray.append("ZERO-BYTE " + p)
json.dump({"inventory": inv, "stray": stray}, open(f"{OUT}/inventory.json", "w"), indent=1)

with open(f"{OUT}/inventory.md", "w") as fh:
    fh.write("| dir | file | ds | n_rows | uniq | n_ref | missing | extra | q_mis | gold_mis | preds min/med/max | exp_rows/expected | judge_cov | j_orph | mtime | flags |\n|" + "---|" * 16 + "\n")
    for r in sorted(inv, key=lambda r: (r["dir"], r["ds"] or "", r["tag"])):
        fl = []
        if r.get("missing_vs_ref"): fl.append("SHORT")
        if r.get("extra_vs_ref"): fl.append("EXTRA")
        if r.get("n_dup_idx_rows"): fl.append("DUPIDX")
        if r.get("q_mismatch") or r.get("gold_mismatch"): fl.append("STALE?")
        if r.get("bad_json"): fl.append("BADJSON")
        if r.get("exp_missing_keys") or r.get("exp_extra_keys") or r.get("exp_text_mismatch"): fl.append("EXPLODE-MISMATCH")
        if r.get("judge_missing_pairs"): fl.append("JUDGE-PARTIAL")
        if r.get("judge_orphans"): fl.append("JUDGE-ORPHAN")
        if r.get("judge_older_than_dump"): fl.append("JUDGE<DUMP")
        if r.get("judge_older_than_exploded"): fl.append("judge<exploded")
        if r.get("djudge_missing"): fl.append("GJUDGE-PARTIAL")
        if r.get("djudge_older_than_dump"): fl.append("GJUDGE<DUMP")
        if r.get("preds_min") != r.get("preds_max"): fl.append("RAGGED-N")
        if r.get("schema_mixed"): fl.append("mixed-schema")
        jc = r.get("judge_cov_pairs", r.get("djudge_cov"))
        fh.write(f"| {r['dir']} | {r['file']} | {r['ds']} | {r['n_rows']} | {r.get('n_unique_idx')} | {r.get('n_ref')} | "
                 f"{r.get('missing_vs_ref')} | {r.get('extra_vs_ref')} | {r.get('q_mismatch')} | {r.get('gold_mismatch')} | "
                 f"{r.get('preds_min')}/{r.get('preds_med')}/{r.get('preds_max')} | {r.get('exp_rows')}/{r.get('exp_expected_rows')} | "
                 f"{('%.4f' % jc) if jc is not None else None} | {r.get('judge_orphans', r.get('djudge_orphans'))} | {r['mtime']} | {' '.join(fl)} |\n")
    fh.write("\nSTRAY / ZERO-BYTE:\n" + "\n".join(stray) + "\n")
tot_q = sum(r.get("n_compared_q", 0) for r in inv)
print("dumps:", len(inv), "| question comparisons vs dataset:", tot_q, "| q mismatches:", sum(r.get("q_mismatch", 0) for r in inv),
      "| gold mismatches:", sum(r.get("gold_mismatch", 0) for r in inv))
print("exploded text comparisons:", sum(r.get("exp_n_text_compared", 0) for r in inv), "mismatches:", sum(r.get("exp_text_mismatch", 0) for r in inv))
print("stray:", len(stray))
