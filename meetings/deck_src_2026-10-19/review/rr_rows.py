import json, hashlib, collections
F = "/home/jamesyang/medvlthinker-imgdiff-compute/feats_hidden"
def half(i): return int(hashlib.md5(("nd" + str(i)).encode()).hexdigest(), 16) % 2
def rows(stem, dsf=None):
    m = json.load(open(f"{F}/{stem}.meta.json"))
    return [r for r in m["rows"] if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
TD = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
tr = rows("generator_train_finelayer", TD)
cells = ['pathvqa_open', 'slake_open', 'vqa_rad_open', 'radimagenet_open', 'kvasir_x1_open', 'omnimed_open',
         'vqamed_open', 'gemex_open', 'vqa_rad_closed', 'pathvqa_closed', 'slake_closed', 'vqamed2019_test']
C = {}
for c in cells:
    C[c] = rows("generator_eval_finelayer", {c}) if c in {"slake_open", "vqa_rad_open"} else rows(f"generator_eval_finelayer_{c}")
print("orig rows", len(tr), collections.Counter(r['ds'] for r in tr))
for f in (0, 1):
    evimg = set()
    for c in cells:
        evimg |= {r['img_md5'] for r in C[c] if half(r['img_md5']) == f}
    o = [r for r in tr if r['img_md5'] not in evimg]
    dropped = collections.Counter(r['ds'] for r in tr if r['img_md5'] in evimg)
    tot = len(o); per = {}
    for c in cells:
        k = [r for r in C[c] if half(r['img_md5']) == 1 - f and r['img_md5'] not in evimg]
        per[c] = len(k); tot += len(k)
    print("fold", f, "evimg", len(evimg), "orig kept", len(o), "orig dropped", dict(dropped), "total", tot)
    print("  per cell train rows", per)
timg = collections.defaultdict(set)
for r in tr: timg[r['ds']].add(r['img_md5'])
print("train domain images", {d: len(s) for d, s in timg.items()})
for c in cells:
    ci = {r['img_md5'] for r in C[c]}
    ov = {d: len(ci & s) for d, s in timg.items() if ci & s}
    print("overlap", c, "images", len(ci), ov)
for i, a in enumerate(cells):
    for b in cells[i + 1:]:
        n = len({r['img_md5'] for r in C[a]} & {r['img_md5'] for r in C[b]})
        if n: print("cell-cell", a, b, n)
print("example row keys", list(tr[0].keys()))
