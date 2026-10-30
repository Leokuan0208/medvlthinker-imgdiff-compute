#!/usr/bin/env python3
"""refit_variance.py -- how much does the final method's headline move if its 24 probes are simply refitted?

Owed since the 2026-10-19 deck ("evaluation owed: refit variance"). Every claim the project makes about a
change to the probe is a difference of ~0.001-0.005 macro, so the run-to-run spread of the shipped recipe is
the yardstick those differences must be read against. Refitting the SAME data with the SAME recipe at the
SAME thread count is bitwise deterministic (CLAUDE.md, 2026-09-13), so the only thing that varies here is the
probes' random initialisation seed.

HOW. Runs final_method_eval.main() UNCHANGED -- same cells (open8 + the four comparability cells, which are
part of the fitting set), same two-fold image cross-fitting, same T07 and Tmap pools, same rank_vote readout,
same currencies and bootstrap -- with one monkeypatch: head_sweep.fit's seed is shifted by REFIT_SEED_OFFSET,
so replicate k fits seeds {k+0 .. k+7} instead of {0 .. 7}. Offset 0 is the shipped run itself
(final_eval_FINAL_cap320_{T07,Tmap}.json); this script never needs to rerun it.

  REFIT_SEED_OFFSET=100 OMP_NUM_THREADS=2 python3 src/exp_2026_10/refit_variance.py
Writes <MAIN>/results/cascade_methods/artifacts/final_eval_FINAL_refit<off>_{T07,Tmap}.json (final_method_eval's
own output path) and /data/dan/exp_2026_10/final_eval_FINAL_refit<off>_*_records.json.
"""
import os, sys
sys.dont_write_bytecode = True
OFF = int(os.environ["REFIT_SEED_OFFSET"])
THREADS = os.environ.get("OMP_NUM_THREADS", "2")
WT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15"
sys.path.insert(0, WT + "/src/exp_2026_10")
import final_method_eval as FME                          # noqa: E402

_fit = FME.HS.fit


def fit_shifted(*a, seed=0, **k):
    return _fit(*a, seed=seed + OFF, **k)


FME.HS.fit = fit_shifted
sys.argv = ["final_method_eval.py", "--name", f"FINAL_refit{OFF}", "--cells", *FME.OPEN8, *FME.COMP,
            "--policy", "T07", "Tmap", "--readout", "rank_vote", "--folds", "0", "1", "--seeds", "8",
            "--threads", THREADS, "--legacy_cap320", "--xjudge", "/data/dan/exp_2026_10/xjudge_c320/merged.jsonl"]
print(f"refit replicate: seeds shifted by {OFF}; argv {sys.argv[1:]}", flush=True)
os.makedirs("/data/dan/exp_2026_10", exist_ok=True)
FME.main()
