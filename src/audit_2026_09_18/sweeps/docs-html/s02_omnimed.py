import json, os, collections
p = '/data/dan/dataset/omnimed_opentext/omnimed_open.json'
print(os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else None)
d = json.load(open(p))
print(type(d), len(d))
it = d[0] if isinstance(d, list) else list(d.values())[0]
print({k: (str(v)[:80]) for k, v in it.items()})
rows = d if isinstance(d, list) else list(d.values())
for key in ('question_type', 'qtype', 'type', 'modality', 'modality_type', 'dataset', 'source'):
    if key in rows[0]:
        c = collections.Counter(r.get(key) for r in rows)
        print(key, len(c), c.most_common(12))
for key in ('gold', 'answer', 'gt_answer'):
    if key in rows[0]:
        golds = set(str(r[key]).strip().lower() for r in rows)
        print('distinct golds (%s, lowercased):' % key, len(golds))
