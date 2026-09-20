#!/usr/bin/env python3
import os, time, collections
R = "/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa"
for d in sorted(os.listdir(R)):
    p = os.path.join(R, d)
    fs = []
    for dp, dn, fn in os.walk(p):
        for f in fn:
            fp = os.path.join(dp, f); fs.append((os.path.getmtime(fp), fp))
    if not fs:
        print(f"{d:40} EMPTY"); continue
    fs.sort()
    ext = collections.Counter("judge" if f.endswith(".judge.jsonl") else "exploded" if "_scexploded" in f else
                              "dump" if os.path.basename(f).startswith("ckpt_") and f.endswith(".jsonl") else "other" for _, f in fs)
    print(f"{d:40} {len(fs):4} files  first {time.strftime('%Y-%m-%d', time.localtime(fs[0][0]))}  "
          f"last {time.strftime('%Y-%m-%d %H:%M', time.localtime(fs[-1][0]))}  {dict(ext)}")
