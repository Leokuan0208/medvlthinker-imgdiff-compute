# Open-text baseline (judge-based) — the comparison anchor for the unified method
Datasets: SLAKE, VQA-RAD, PathVQA, Kvasir, RadImageNet (open-ended free-text, LLM-judge). RadImageNet = held-out OOD.
Generated 2026-06-29 from ckpts/openvqa (our NGC pipeline). greedy = 1st sample; SC = majority; oracle@8 = any-of-8 correct.

| model | SLAKE | VQA-RAD | PathVQA | Kvasir | RadImageNet |
| Lingshu-7B greedy | 0.722 | 0.420 | 0.295 | 0.287 | 0.321 |
| Lingshu-7B SC | 0.736 | 0.460 | 0.321 | 0.287 | 0.328 |
| Lingshu-7B oracle@8 | 0.879 | 0.630 | 0.517 | 0.491 | 0.512 |
| Lingshu-32B | 0.819 | 0.600 | 0.376 | 0.301 | (gap) |
| MedVLThinker-7B greedy | 0.543 | 0.395 | (gap) | (gap) | (gap) |
| MedVLThinker-32B | (gap) | (gap) | (gap) | (gap) | (gap) |

POOLED (4 in-dist, n-weighted): Lingshu-7B greedy 0.377, 32B 0.444, oracle@8 ~0.62. (Matches prior verifier work.)
KEY INSIGHT: verifier(7B,bo8) BEATS 32B on PathVQA+Kvasir (32B weak: 0.38/0.30, oracle 0.52/0.49), LOSES on
SLAKE+VQA-RAD (32B strong 0.82/0.60). => the UNIFIED method's value is ROUTING: 7B+verifier where it wins, escalate
to 32B where it doesn't -> can beat pooled-32B (0.444) at <100% escalation (lower FLOPs). This is the feasible, measurable win.
GAPS TO FILL (GPU, our pipeline): Lingshu-32B@radimagenet; MedVLThinker-7B@{pathvqa-judge,kvasir,radimagenet};
MedVLThinker-32B@all5+judge. Stretch: InternVL3-8B/38B (needs download + baseline).
