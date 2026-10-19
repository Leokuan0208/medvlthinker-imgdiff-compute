"""Round-2 audit helper: print selected keys of artifacts (read-only). Usage: r2_peek.py file [dotted.key ...] [--n N]"""
import json, sys
A = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts/"
M = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/"


def r(x):
    if isinstance(x, float): return round(x, 4)
    if isinstance(x, list): return [r(y) for y in x]
    if isinstance(x, dict): return {k: r(v) for k, v in x.items()}
    return x


args = sys.argv[1:]
n = 3000
if "--n" in args:
    i = args.index("--n"); n = int(args[i + 1]); args = args[:i] + args[i + 2:]
f = args[0]
try:
    d = json.load(open(f if f.startswith("/") else A + f))
except FileNotFoundError:
    d = json.load(open(M + f))
keys = args[1:] or [""]
for k in keys:
    v = d
    for p in [p for p in k.split(".") if p]:
        v = v[p] if not isinstance(v, list) else v[int(p)]
    if isinstance(v, dict) and k == "":
        print("KEYS", list(v.keys()))
    print(k, json.dumps(r(v))[:n])
