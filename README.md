# medvlthinker-imgdiff-compute

Question-Aware Adaptive Compute for Medical VLMs — image-difficulty-driven
reasoning-budget allocation. Base model: MedVLThinker-3B-RL_m23k (Qwen2.5-VL).

## Lineage
- huatuo-llava-v15-med-pruning  : 2D question-aware token pruning (frozen; random>=QSim dead end)
- medvlthinker-imgdiff-compute  : PIVOT. Hypothesis: image content (not the question)
  drives per-case difficulty; allocate reasoning compute on that signal.

## Pipeline (gate)
1. build_subset.py            -> subset.csv  (SLAKE yes/no closed, stratified)
2. difficulty_medvlthinker.py -> difficulty.csv  (pass-count difficulty on the 3B; = training labels)
3. complexity.py              -> complexity.csv  (question-free image complexity)
4. complexity_lesion.py       -> adds comp_lesion_* (REFINE: SLAKE organ-mask region size/contrast)
5. analyze.py                 -> GO / REFINE / NO-GO  (difficulty~complexity | question_type+modality)

## Status
- Track-1 (HuatuoGPT) gate: REFINE — image->difficulty real & significant but weak (|rho|<=0.11),
  NEGATIVE sign (busier=easier) => whole-image texture likely measures evidence-richness,
  not lesion subtlety. Running definitive 3B gate + lesion-aware refinement before training.
