"""Flatten every artifact JSON (+ recipe.json) into (file, path, value) leaves. Read-only on the repo."""
import json, glob, os, pickle, sys
ROOT = '/home/jamesyang/medvlthinker-imgdiff-compute'
files = sorted(glob.glob(ROOT + '/results/cascade_methods/artifacts/*.json'))
files += sorted(glob.glob(ROOT + '/ckpts/train/genframe_head_pooled_ens*/recipe.json'))
leaves = []
def walk(o, path, f):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, path + '/' + str(k), f)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, path + '[%d]' % i, f)
    else:
        leaves.append((f, path, o))
for f in files:
    try:
        if os.path.getsize(f) > 60e6:
            print('skip big', f); continue
        d = json.load(open(f))
    except Exception as e:
        print('ERR', f, e); continue
    walk(d, '', os.path.relpath(f, ROOT))
pickle.dump(leaves, open('/data/dan/audit_2026-09-18/tmp/docs-html/leaves.pkl', 'wb'))
print(len(files), 'files', len(leaves), 'leaves')
