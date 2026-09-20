import json, os, re, sys
A='/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/'
names=["head_final_stack_PVFIXED_2026-09-13.json","head_final_stack_qwen_2026-09-13.json",
"head_final_stack_medgemma_ALL8_2026-09-16.json","head_lobo_pooled_2026-08-25.json",
"head_price_from_lobo_2026-08-30.json","coverage_sc16_ci_ALL_2026-09-16.json",
"head_temp_ensemble_2026-08-30.json","decomposition_2026-08-24.json",
"free_signal_bakeoff_2026-08-21.json","head_arch_transfer_2026-08-19.json",
"tiebreak_2026-09-13.json","repro_threading_2026-09-13.json","head_pooled_alldomains_2026-08-24.json",
"budget_conversion_sc32_2026-09-17.json","head_final_stack_2026-08-24.json"]
def walk(o,p="",depth=0,out=None):
    if out is None: out=[]
    if depth>3: return out
    if isinstance(o,dict):
        for k,v in o.items():
            walk(v,p+"/"+str(k),depth+1,out)
    elif isinstance(o,list):
        out.append((p+"[]",f"list len {len(o)}", str(o[:2])[:200]))
    else:
        out.append((p,type(o).__name__,str(o)[:160]))
    return out
for n in names:
    fp=A+n
    if not os.path.exists(fp):
        print("### MISSING",n); continue
    d=json.load(open(fp))
    print("="*100); print("###",n, "size",os.path.getsize(fp))
    print("  TOP KEYS:", list(d.keys())[:40] if isinstance(d,dict) else f"list {len(d)}")
    s=json.dumps(d)
    for pat in ["ci","seed","bootstrap","cluster","n_boot","threads","argv","git","verdict","VERDICT"]:
        hits=sorted(set(re.findall(r'"([^"]*%s[^"]*)"\s*:'%pat, s, re.I)))[:12]
        if hits: print(f"   keys~{pat}: {hits}")
    # print shallow scalars
    for p,t,v in walk(d)[:60]:
        print("   ", p, "=", v)
