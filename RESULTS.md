# Results index — MedVLThinker efficiency cascade

## Canonical benchmark names (use everywhere)
| parquet `dataset_name` | our name | paper column | n (closed subset) |
|---|---|---|---|
| pmc_vqa         | PMC-VQA  | PMC      | 2000 |
| MMMU-medical    | MMMU     | MMMU     | 170  |
| MedXpertQA-MM   | MedX-M   | MedX-M   | 2000 (= our R 1446 + U 554) |
| pathvqa_closed  | PathVQA  | PathVQA  | 3362 |
| slake_closed    | SLAKE    | SLAKE    | 416  |
| vqa_rad_closed  | VQA-RAD  | VQA-Rad  | 272  |

## Result folders (current layout)
- ckpts/gate_7b_vllm/            7B no_think, full-res, EVAL    (all 6)
- ckpts/gate_7b_prune/cap{640,320,160,80}/  7B no_think, capped, EVAL  (4 competent; MISSING MedXpert+MMMU)
- ckpts/gate_32b/               32B think, full-res, EVAL       (all 6)
- archive/single-model-routing/gate_7b_rag_axes/  (was ckpts/gate_7b_v2) 7B n~500 grid over think/nothink/RAG axes — KILLED RAG experiment, archived
- ckpts/gate_7b_pmctrain/       7B no_think, full-res, CALIB (PMC-train, 3000)
- ckpts/gate_7b_pmctrain_prune/cap{...}/  7B no_think, capped, CALIB (3000 each)
- ckpts/router_margin.pkl       frozen margin gate, tau=0.426

## File naming
- EVAL  files: ckpt_<benchmark>_<mode>_norag_s<k>of<N>.jsonl   (mode = nothink | think)
- CALIB files: ckpt_<mode>_s<k>of<N>.jsonl
- row schema: {idx, gold, pred, ok, parse_ok, opt_logprobs{A..}, gen_tokens, raw_output}

## Proposed clean layout (when we migrate, after the remaining runs land)
runs/eval/<model>_<mode>/<resolution>/     e.g. runs/eval/7b_nothink/cap320/
runs/calib/7b_nothink/<resolution>/        (PMC-train, for tau)
runs/artifacts/router_margin.pkl
  resolution in {fullres, cap640, cap320, cap160, cap80}

## Validation status (2026-06-10)
- Harness CORRECT: 7B-think and 32B-think both reproduce the paper (PMC, MedX-M) within ~1pt.
- Subsets = paper's CLOSED subsets (vqa_rad 272, slake 416, pathvqa 3362).
- no_think = a faster mode, ~+2-3pt on perception sets, ~flat on MedXpert.

## Known gaps (Task 3)
- MMMU: data present (MMMU-medical, 170) but no DATASET_IDX entry; never run on either model.
- MedXpert: missing at all caps (only full-res exists).
- 7B think: only PMC + MedXpert at n~500; missing SLAKE/VQA-RAD/PathVQA entirely.
