# Read-only: compare STRUCTURE.md and results/cascade_methods/README.md against the tree on disk (MAIN).
import os, re, collections, json
M = '/home/jamesyang/medvlthinker-imgdiff-compute'
OUT = '/data/dan/audit_2026-09-18/tmp/docs-top/'

def walk(root, exts):
    out = []
    for dp, dn, fn in os.walk(os.path.join(M, root)):
        dn[:] = [d for d in dn if d not in ('__pycache__', 'graphify-out', '.ipynb_checkpoints')]
        for f in fn:
            if f.endswith(exts):
                out.append(os.path.relpath(os.path.join(dp, f), M))
    return sorted(out)

py = walk('src', ('.py',))
sh = [p for p in walk('runners', ('.sh',))]
print('src .py on disk:', len(py))
cnt = collections.Counter(p.split('/')[1] if p.count('/') >= 2 else '<src root>' for p in py)
for k, v in sorted(cnt.items()): print('   ', k, v)
print('runners .sh on disk:', len(sh))
print('runners total entries:', len(os.listdir(os.path.join(M, 'runners'))))

S = open(os.path.join(M, 'STRUCTURE.md'), encoding='utf-8').read()

def mentioned(path, text):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    # exact basename mention, or stem as a whole token (STRUCTURE.md sometimes drops the .py)
    if base in text: return True
    if re.search(r'(?<![A-Za-z0-9_])' + re.escape(stem) + r'(?![A-Za-z0-9_])', text): return True
    return False

miss_py = [p for p in py if not mentioned(p, S)]
miss_sh = [p for p in sh if not mentioned(p, S)]
print('src .py NOT mentioned in STRUCTURE.md:', len(miss_py), 'of', len(py))
print('runners .sh NOT mentioned in STRUCTURE.md:', len(miss_sh), 'of', len(sh))
cm = collections.Counter(p.split('/')[1] for p in miss_py)
print('   by dir:', dict(cm))
open(OUT + 'structure_missing_py.txt', 'w').write('\n'.join(miss_py) + '\n')
open(OUT + 'structure_missing_sh.txt', 'w').write('\n'.join(miss_sh) + '\n')

# files STRUCTURE.md mentions that no longer exist
toks = set(re.findall(r'[A-Za-z0-9_\-./{},*]+\.(?:py|sh|md|json|html|pdf|tex|pkl|jsonl|docx|txt|csv)', S))
allfiles = set()
base_index = collections.defaultdict(list)
for dp, dn, fn in os.walk(M):
    rel = os.path.relpath(dp, M)
    if rel.startswith('.git') or rel.startswith('.claude') or '__pycache__' in rel:
        dn[:] = []; continue
    # do not descend into the giant data dirs
    if rel.split('/')[0] in ('feats', 'feats_full', 'feats_hidden', 'feats_hidden_noise', 'feats_peer', 'feats_vision', 'feats_free', 'data', 'tools', 'graphify-out'):
        dn[:] = []; continue
    if rel.startswith('MedEvalKit/eval_results') or rel.startswith('MedRAG/corpus') or rel.startswith('src/graphify-out'):
        dn[:] = []; continue
    for f in fn:
        p = os.path.normpath(os.path.join(rel, f))
        allfiles.add(p); base_index[f].append(p)
gone = []
for t in sorted(toks):
    if any(c in t for c in '{}*'): continue
    t2 = t.lstrip('./')
    b = os.path.basename(t2)
    if t2 in allfiles: continue
    if '/' in t2:
        # path given: accept if some file ends with that path
        if any(p.endswith(t2) for p in base_index.get(b, [])): continue
        gone.append((t, 'path not found; basename exists at: ' + ', '.join(base_index[b][:2]) if b in base_index else 'NOT FOUND ANYWHERE'))
    else:
        if b in base_index: continue
        gone.append((t, 'NOT FOUND ANYWHERE'))
print('STRUCTURE.md file tokens:', len(toks), ' unresolved:', len(gone))
with open(OUT + 'structure_mentions_missing.txt', 'w') as fh:
    for t, why in gone: fh.write(f'{t}\t{why}\n')
for t, why in gone: print('   ', t, '|', why[:110])

# results/cascade_methods/README.md index coverage
R = open(os.path.join(M, 'results/cascade_methods/README.md'), encoding='utf-8').read()
docs = sorted(f for f in os.listdir(os.path.join(M, 'results/cascade_methods/docs/current')) if f.endswith(('.md', '.html', '.docx')))
nd = [d for d in docs if d not in R]
print('docs/current files:', len(docs), ' NOT indexed in results README:', len(nd))
for d in nd: print('   ', d)
arts = sorted(f for f in os.listdir(os.path.join(M, 'results/cascade_methods/artifacts')) if f.endswith('.json'))
na = [a for a in arts if a not in R and os.path.splitext(a)[0] not in R]
print('top-level artifact .json:', len(arts), ' NOT indexed in results README:', len(na))
sept = [a for a in arts if re.search(r'2026-09-\d\d', a)]
print('September-dated artifacts:', len(sept), ' of which indexed:', sum(1 for a in sept if a in R))
bymonth = collections.Counter((re.search(r'2026-(\d\d)-\d\d', a) or [None, 'undated'])[1] if re.search(r'2026-(\d\d)-\d\d', a) else 'undated' for a in arts)
print('artifact .json by month tag:', dict(bymonth))
bym_idx = collections.Counter((re.search(r'2026-(\d\d)-\d\d', a).group(1) if re.search(r'2026-(\d\d)-\d\d', a) else 'undated') for a in arts if a in R or os.path.splitext(a)[0] in R)
print('indexed artifact .json by month tag:', dict(bym_idx))
open(OUT + 'results_readme_unindexed_artifacts.txt', 'w').write('\n'.join(na) + '\n')
open(OUT + 'results_readme_sept_artifacts.txt', 'w').write('\n'.join(sept) + '\n')
# artifacts the README names that are not on disk
rtoks = set(re.findall(r'[A-Za-z0-9_\-]+\.json', R))
rg = sorted(t for t in rtoks if t not in base_index)
print('results README .json tokens:', len(rtoks), ' not found on disk:', len(rg), rg[:40])
rmd = set(re.findall(r'[A-Za-z0-9_\-]+\.md', R))
rgm = sorted(t for t in rmd if t not in base_index)
print('results README .md tokens:', len(rmd), ' not found on disk:', rgm)
