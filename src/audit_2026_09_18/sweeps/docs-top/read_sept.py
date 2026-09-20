import json,os
A='/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/'
for f in ['head_final_stack_qwen_2026-09-13.json','head_final_stack_medgemma_ALL8_2026-09-16.json']:
    d=json.load(open(A+f)); print('#####',f)
    print({k:d[k] for k in d if k not in('cells',)})
    print(' n total',sum(c['n_questions'] for c in d['cells'].values()), {k:c['n_questions'] for k,c in d['cells'].items()})
for f in ['coverage_sc16_ci_ALL_2026-09-16.json','repro_threading_2026-09-13.json']:
    d=json.load(open(A+f)); print('#####',f, os.path.getsize(A+f))
    for k,v in d.items(): print('  ',k,':',json.dumps(v)[:700])
