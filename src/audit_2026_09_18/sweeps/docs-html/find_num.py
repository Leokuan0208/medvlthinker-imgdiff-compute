"""find_num.py [-f filefilter] num [num ...]
For each printed number (e.g. +0.0736, 44.1%, 0.686, 1,623), list artifact leaves that round to it.
"""
import pickle, sys, re
leaves = pickle.load(open('/data/dan/audit_2026-09-18/tmp/docs-html/leaves.pkl', 'rb'))
args = sys.argv[1:]
flt = None
maxn = 12
while args and args[0] in ('-f', '-m'):
    if args[0] == '-f': flt = args[1]
    else: maxn = int(args[1])
    args = args[2:]
for a in args:
    s = a.replace(',', '').replace('+', '').replace('×', '').replace('x', '')
    pct = s.endswith('%')
    s = s.rstrip('%')
    neg = s.startswith('-') or s.startswith('−')
    s = s.lstrip('-−')
    nd = len(s.split('.')[1]) if '.' in s else 0
    target = float(s)
    hits = []
    for f, p, v in leaves:
        if flt and flt not in f: continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            if isinstance(v, str) and a in v:
                hits.append((f, p, v[:160]))
            continue
        for scale in ((100.0,) if pct else (1.0,)):
            vv = v * scale
            if abs(round(abs(vv), nd) - target) < 10 ** (-nd) / 1000.0 + 1e-12:
                if neg and vv > 0: continue
                hits.append((f, p, v))
    print('=== %s : %d hits' % (a, len(hits)))
    for h in hits[:maxn]:
        print('   ', h[0].replace('results/cascade_methods/artifacts/', ''), h[1], h[2])
