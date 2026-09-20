"""READ-ONLY integrity check of generation dumps, exploded files, judge files for the 8 benchmarks x 3 generators.
Checks: row counts vs cell size, duplicate idx, #preds, question-text match (cell json vs sc8 dump vs greedy dump),
exploded<->judge coverage, exploded row content vs dump preds[k], greedy==candidate string judge agreement,
EM(oks) vs judge agreement on candidates."""
import json, os, sys, glob, io
from collections import Counter, defaultdict
CK = "/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b"
CELLS = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
TAGS = ["lingshu7b", "qwen25vl7b", "medgemma4b"]
JSON_CELLS = {"radimagenet_open": "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json",
              "kvasir_x1_open": "/data/dan/dataset/kvasir_x1_cell/kvasir_x1_open.json",
              "omnimed_open": "/data/dan/dataset/omnimed_opentext/omnimed_open.json",
              "vqamed_open": "/data/dan/dataset/vqamed_cell/vqamed_open.json",
              "gemex_open": "/data/dan/dataset/gemex_cell/gemex_open.json"}

def loadl(p):
    out = []
    if not os.path.exists(p): return None
    bad = 0
    for l in open(p):
        if l.strip():
            try: out.append(json.loads(l))
            except Exception: bad += 1
    if bad: print(f"   !! {os.path.basename(p)}: {bad} unparseable lines")
    return out

def cell_questions(cell):
    if cell in JSON_CELLS:
        d = json.load(open(JSON_CELLS[cell]))
        return {r["idx"]: (r["question"], str(r["answer"])) for r in d}
    if cell == "slake_open":
        d = json.load(open("/data/dan/dataset/slake/test.json"))
        m = {}
        for x in d:
            if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en" and os.path.exists(os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])):
                m[x["qid"]] = (x["question"], str(x["answer"]))
        return m
    import pandas as pd
    base = "/data/dan/dataset/vqa_rad/data" if cell == "vqa_rad_open" else "/data/dan/dataset/path_vqa/data"
    fs = sorted(glob.glob(f"{base}/test-*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=[c for c in ["question", "answer"]]) for f in fs], ignore_index=True)
    m = {}
    for i, r in df.iterrows():
        a = str(r["answer"]).strip()
        if a.lower() in ("yes", "no"): continue
        m[int(i)] = (str(r["question"]), a)
    return m

def nrm(s): return str(s).strip().lower()

for cell in CELLS:
    try:
        CQ = cell_questions(cell)
    except Exception as e:
        print(cell, "cell load failed", e); CQ = None
    print(f"\n===== {cell}: cell size {len(CQ) if CQ else 'NA'}")
    for tag in TAGS:
        sc = loadl(f"{CK}/ckpt_{cell}_{tag}_sc8.jsonl")
        gr = loadl(f"{CK}/ckpt_{cell}_{tag}.jsonl")
        gj = loadl(f"{CK}/ckpt_{cell}_{tag}.judge.jsonl")
        ex = loadl(f"{CK}/ckpt_{cell}_{tag}_sc8_scexploded.jsonl")
        ej = loadl(f"{CK}/ckpt_{cell}_{tag}_sc8_scexploded.judge.jsonl")
        if sc is None: print(f"  [{tag}] no sc8 dump"); continue
        scd = {r["idx"]: r for r in sc}
        print(f"  [{tag}] sc8 rows {len(sc)} uniq {len(scd)} | preds!=8: {sum(1 for r in sc if len(r['preds'])!=8)} | has 'question' {sum(1 for r in sc if r.get('question') is not None)}")
        if CQ is not None:
            miss = [i for i in CQ if i not in scd]; extra = [i for i in scd if i not in CQ]
            qmis = sum(1 for i, r in scd.items() if i in CQ and str(r.get("question")).strip() != str(CQ[i][0]).strip())
            gmis = sum(1 for i, r in scd.items() if i in CQ and str(r.get("gold")).strip() != str(CQ[i][1]).strip())
            print(f"      vs cell: missing-from-dump {len(miss)} extra-in-dump {len(extra)} question-text-mismatch {qmis} gold-mismatch {gmis}")
        if gr is not None:
            grd = {r["idx"]: r for r in gr}
            print(f"      greedy rows {len(gr)} uniq {len(grd)} | idx only-in-sc8 {len(set(scd)-set(grd))} only-in-greedy {len(set(grd)-set(scd))} | q-mismatch sc8-vs-greedy {sum(1 for i in grd if i in scd and str(grd[i].get('question')).strip()!=str(scd[i].get('question')).strip())}")
            if gj is not None:
                gjd = {}
                dup = 0
                for r in gj:
                    if r["idx"] in gjd: dup += 1
                    gjd.setdefault(r["idx"], r["judge_ok"])
                print(f"      greedy judge rows {len(gj)} dup {dup} | greedy idx unjudged {len(set(grd)-set(gjd))} | judge idx not in greedy dump {len(set(gjd)-set(grd))}")
        if ex is not None and ej is not None:
            exd = {r["idx"]: r for r in ex}
            ejd = {}
            dup = 0
            for r in ej:
                if r["idx"] in ejd: dup += 1
                ejd.setdefault(r["idx"], r["judge_ok"])
            unj = set(exd) - set(ejd); orph = set(ejd) - set(exd)
            # exploded content vs dump
            stale = 0; qstale = 0; nomatch = 0
            for cid, r in exd.items():
                oi, k = cid.rsplit("#", 1); k = int(k)
                oi2 = int(oi) if oi.lstrip("-").isdigit() else oi
                d = scd.get(oi2, scd.get(oi))
                if d is None: nomatch += 1; continue
                if k >= len(d["preds"]) or d["preds"][k] != r["modal_pred"]: stale += 1
                if str(d.get("question")).strip() != str(r.get("question")).strip(): qstale += 1
            # expected distinct
            exp_n = sum(len({nrm(a) for a in r["preds"]}) for r in sc)
            qs_ex = {cid.rsplit("#", 1)[0] for cid in exd}
            print(f"      exploded rows {len(ex)} (expected from dump {exp_n}) questions {len(qs_ex)} | judge rows {len(ej)} dup {dup} | exploded-unjudged {len(unj)} judge-orphans {len(orph)} | exploded!=dump.preds[k] {stale} q-mismatch {qstale} idx-not-in-dump {nomatch}")
            print(f"      mean distinct cands/question {exp_n/len(sc):.3f}")
            # greedy string == candidate string: judge agreement
            if gr is not None and gj is not None:
                agree = dis = 0
                lab = defaultdict(dict)
                for cid, r in exd.items():
                    if cid in ejd:
                        lab[cid.rsplit("#", 1)[0]][nrm(r["modal_pred"])] = ejd[cid]
                for i, g in grd.items():
                    if i in gjd and str(i) in lab:
                        a = nrm(g["modal_pred"])
                        if a in lab[str(i)]:
                            if lab[str(i)][a] == gjd[i]: agree += 1
                            else: dis += 1
                print(f"      greedy-string-in-pool: {agree+dis} questions, judge label disagreement {dis} ({(dis/max(agree+dis,1))*100:.2f}%)")
            # EM vs judge over candidates
            n11 = n10 = n01 = n00 = 0
            for i, r in scd.items():
                seen = set()
                for k, (a, ok) in enumerate(zip(r["preds"], r["oks"])):
                    na = nrm(a)
                    if na in seen: continue
                    seen.add(na)
                    cid = f"{i}#{k}"
                    if cid in ejd:
                        j = ejd[cid]
                        if j and ok: n11 += 1
                        elif j and not ok: n10 += 1
                        elif ok and not j: n01 += 1
                        else: n00 += 1
            tot = n11 + n10 + n01 + n00
            if tot:
                print(f"      distinct cands: judge=1&EM=1 {n11} | judge=1&EM=0 {n10} ({n10/tot*100:.1f}%) | judge=0&EM=1 {n01} ({n01/tot*100:.1f}%) | both0 {n00} | judge-pos-rate {(n11+n10)/tot:.4f} EM-pos-rate {(n11+n01)/tot:.4f}")
