#!/usr/bin/env python3
"""Artifact finder. Read-only. Usage:
  af.py find <value> [artifact-substring] [--abs]  -> list every leaf equal to value at printed precision
  af.py keys <artifact-substring> [depth]          -> show structure
  af.py get <artifact-substring> <regex-on-dotted-key>
"""
import json, sys, os, glob, re
MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
ARTDIRS = [MAIN + "/results/cascade_methods/artifacts"]


def flatten(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flatten(v, f"{pre}[{i}]")
    else:
        yield pre, o


def files(sub=None):
    fs = []
    for d in ARTDIRS:
        fs += sorted(glob.glob(d + "/*.json"))
    fs += sorted(glob.glob(MAIN + "/ckpts/train/genframe_head_pooled_ens*/recipe.json"))
    if sub:
        fs = [f for f in fs if sub in f]
    return fs


def match(printed, val):
    s = printed.replace(",", "").replace("+", "").replace("−", "-").replace("%", "")
    try:
        p = float(s)
    except ValueError:
        return False
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    dec = len(s.split(".")[1]) if "." in s else 0
    cands = [val]
    if "%" in printed:
        cands.append(val * 100)
    for cand in cands:
        if abs(cand - p) <= 0.5 * 10 ** (-dec) + 1e-9:
            return True
    return False


if __name__ == "__main__":
    cmd = sys.argv[1]
    args = [a for a in sys.argv[2:] if a != "--abs"]
    absm = "--abs" in sys.argv
    if cmd == "find":
        val = args[0]
        sub = args[1] if len(args) > 1 else None
        for f in files(sub):
            try:
                o = json.load(open(f))
            except Exception:
                continue
            for k, v in flatten(o):
                if match(val, v) or (absm and isinstance(v, (int, float)) and not isinstance(v, bool) and match(val, -v)):
                    print(f"{os.path.basename(f) if 'artifacts' in f else f}\t{k}\t{v}")
    elif cmd == "keys":
        sub = args[0]
        depth = int(args[1]) if len(args) > 1 else 2
        for f in files(sub):
            print("==", f, os.path.getsize(f))
            o = json.load(open(f))

            def show(o, d, ind):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if isinstance(v, (dict, list)) and d < depth:
                            print("  " * ind + str(k) + (f" [list{len(v)}]" if isinstance(v, list) else ""))
                            show(v, d + 1, ind + 1)
                        else:
                            sv = json.dumps(v)
                            print("  " * ind + f"{k}: {sv[:200]}")
                elif isinstance(o, list):
                    for i, v in enumerate(o[:3]):
                        print("  " * ind + f"[{i}]")
                        show(v, d + 1, ind + 1)
            show(o, 0, 1)
    elif cmd == "get":
        sub = args[0]
        path = args[1]
        for f in files(sub):
            o = json.load(open(f))
            for k, v in flatten(o):
                if re.search(path, k):
                    print(f"{os.path.basename(f)}\t{k}\t{v}")
