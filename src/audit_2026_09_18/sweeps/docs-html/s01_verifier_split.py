import json
A = '/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/'
d = json.load(open(A + 'cost_decomposition_2026-08-12.json'))
pc = d['Q0_cost_decomposition']['shipped_accuracy_max_as_charged']['per_cell']
tot = {'verifier_vision': 0, 'verifier_lm_prefill': 0, 'verifier_decode': 0}
for c, x in pc.items():
    st = x['stages']
    v = {k: st.get(k, 0) for k in tot}
    s = sum(v.values())
    if s > 0:
        print(c, {k: round(val / s * 100, 2) for k, val in v.items()}, 'sum', round(s, 4))
    for k in tot:
        tot[k] += v[k]
s = sum(tot.values())
print('ALL', {k: round(val / s * 100, 2) for k, val in tot.items()})
