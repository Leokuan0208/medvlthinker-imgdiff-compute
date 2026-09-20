# Read-only emulation of `git status` for MAIN: compare working-tree blob hashes with the `main` tree.
import os,hashlib,sys
M='/home/jamesyang/medvlthinker-imgdiff-compute'
tracked={}
for l in open('/data/dan/audit_2026-09-18/tmp/docs-top/main_lstree.txt'):
    meta,path=l.rstrip('\n').split('\t',1)
    mode,typ,sha=meta.split()
    tracked[path]=(mode,sha)
mod=[];missing=[]
for p,(mode,sha) in tracked.items():
    fp=os.path.join(M,p)
    if mode=='120000':
        if not os.path.islink(fp): missing.append(p); continue
        data=os.readlink(fp).encode()
    else:
        if not os.path.isfile(fp): missing.append(p); continue
        data=open(fp,'rb').read()
    h=hashlib.sha1(b'blob %d\0'%len(data)+data).hexdigest()
    if h!=sha: mod.append(p)
print('tracked',len(tracked),'modified',len(mod),'missing',len(missing))
for p in mod: print(' M',p)
for p in missing: print(' D',p)
# untracked code/doc files in the dirs that matter (not gitignore-aware; listed by dir)
import collections
unt=collections.defaultdict(list)
for root in ['src','runners','progress','meetings','paper','docx','results/cascade_methods/docs','results/cascade_methods/artifacts','literature']:
    for dp,dn,fn in os.walk(os.path.join(M,root)):
        dn[:]=[d for d in dn if d!='__pycache__']
        for f in fn:
            rel=os.path.relpath(os.path.join(dp,f),M)
            if rel not in tracked and not f.endswith('.pyc'):
                unt[root].append(rel)
for f in sorted(os.listdir(M)):
    if os.path.isfile(os.path.join(M,f)) and f not in tracked: unt['<root>'].append(f)
for k,v in unt.items():
    print('UNTRACKED in',k,len(v))
    for x in sorted(v)[:40]: print('   ',x)
