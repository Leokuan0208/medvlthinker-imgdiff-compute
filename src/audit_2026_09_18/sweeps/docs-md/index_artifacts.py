#!/usr/bin/env python3
"""Flatten all artifact JSONs (+ recipe.json) into a single {file:key -> value} index.
Prints nothing by default; import and use build_index(). Also runnable as CLI dump.
"""
import json, os, glob

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


def files():
    fs = []
    for d in ARTDIRS:
        fs += sorted(glob.glob(d + "/*.json"))
    fs += sorted(glob.glob(MAIN + "/ckpts/train/genframe_head_pooled_ens_v2/recipe.json"))
    return fs


def build_index():
    idx = {}  # (basename, dotted_key) -> value
    for f in files():
        try:
            o = json.load(open(f))
        except Exception as e:
            print("SKIP", f, e)
            continue
        bn = os.path.basename(f) if "artifacts" in f else f.replace(MAIN + "/", "")
        for k, v in flatten(o):
            idx[(bn, k)] = v
    return idx


if __name__ == "__main__":
    idx = build_index()
    print(f"{len(files())} files, {len(idx)} leaves")
    for f in files():
        print(f)
