#!/usr/bin/env python3
"""Recover the generation config of every dump in ckpts/openvqa/cheap_lingshu7b from the runner JSONs
(runners/auto_*_wave*.json), runner shell scripts and logs. READ-ONLY. Output out/runner_configs.json."""
import json, os, re, glob, shlex, time
MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
OUT = "/data/dan/audit_2026-09-18/tmp/data-integrity/out"
cmds = []
for p in sorted(glob.glob(f"{MAIN}/runners/*.json")):
    try: d = json.load(open(p))
    except Exception as e:
        print("unreadable", p, e); continue
    jobs = d if isinstance(d, list) else d.get("jobs", [])
    for j in jobs:
        c = j.get("cmd", "") if isinstance(j, dict) else ""
        if "run_openvqa.py" in c:
            cmds.append((os.path.basename(p), os.path.getmtime(p), j.get("name"), c, j.get("log")))
for p in sorted(glob.glob(f"{MAIN}/runners/*.sh")):
    txt = open(p, errors="ignore").read()
    if "run_openvqa.py" in txt:
        # join continuation lines
        t = txt.replace("\\\n", " ")
        for line in t.splitlines():
            if "run_openvqa.py" in line and not line.strip().startswith("#"):
                cmds.append((os.path.basename(p), os.path.getmtime(p), None, line.strip(), None))

def arg(c, name, default=None):
    m = re.search(r"--%s[ =]+(\"[^\"]*\"|'[^']*'|\S+)" % name, c)
    return m.group(1).strip("'\"") if m else default
rows = []
for src, mt, name, c, log in cmds:
    # a cmd can hold several run_openvqa invocations chained with && or ;
    for part in re.split(r"&&|;|\|\|", c):
        if "run_openvqa.py" not in part: continue
        rows.append({"src": src, "src_mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)), "job": name,
                     "dataset": arg(part, "dataset"), "tag": arg(part, "tag"), "model_path": arg(part, "model_path"),
                     "n_samples": arg(part, "n_samples", "1(default)"), "temp": arg(part, "temp", "0.0(default)"),
                     "cap": arg(part, "cap", "cap320(default)"), "n": arg(part, "n", "100000(default)"),
                     "max_tokens": arg(part, "max_tokens", "64(default)"), "ckpt_dir": arg(part, "ckpt_dir"),
                     "tp": arg(part, "tp", "1(default)"), "idx_file": arg(part, "idx_file"),
                     "gpu_mem": arg(part, "gpu_mem", "0.88(default)"), "max_model_len": arg(part, "max_model_len", "8192(default)"),
                     "flags": [f for f in ("--think", "--think_matched", "--direct_unstyled", "--save_raw") if f in part],
                     "python": ("medeval_venv" if "medeval_venv" in c else "python3"),
                     "log": log, "raw": part.strip()[:600]})
json.dump(rows, open(f"{OUT}/runner_configs.json", "w"), indent=1)
print(len(rows), "run_openvqa invocations found")
import collections
by = collections.defaultdict(list)
for r in rows:
    if r["ckpt_dir"] and "cheap_lingshu7b" in r["ckpt_dir"] and "cheap_lingshu7b_" not in r["ckpt_dir"]:
        by[(r["dataset"], r["tag"])].append(r)
for k in sorted(by, key=lambda t: (str(t[0]), str(t[1]))):
    for r in by[k]:
        print(f"{str(k[0]):22} {str(k[1]):22} T={r['temp']:14} N={r['n_samples']:11} cap={r['cap']:16} n={r['n']:16} maxtok={r['max_tokens']:12} "
              f"model=...{str(r['model_path'])[-46:]} | {r['src']} {r['src_mtime']}")
