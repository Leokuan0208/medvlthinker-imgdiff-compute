#!/bin/bash
# ATTACK 2 phase 3 -- CPU-only assembly of the trained arms, then the one artifact.
# Safe to re-run at any time: every stage reads whatever score files exist and writes "not measured"
# for what does not.  Launch from the repo root.
#
#   bash runners/run_unified_armB_analyse.sh
set -u
REPO=/home/jamesyang/medvlthinker-imgdiff-compute
cd "$REPO"
export HF_HOME=/data/dan/hf_cache OMP_NUM_THREADS=1 PYTHONHASHSEED=0
L="$REPO/logs/unified_armB_analyse.log"
say(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$L"; }

# arm B0 is FORMAT-SPECIFIC (option branch only), so its 8-cell macro keeps the incumbent open
# branch and is labelled as a two-adapter number, not a unified pipeline.
say "== arm B0 (option-only) =="
python3 src/cascade_methods/unified_pipeline.py --analyse --tag optiononly_s0 >>"$L" 2>&1
say "   analyse rc=$?"

# arm B IS the unified pipeline: if its own open dumps exist, the whole 8-cell macro comes from ONE
# adapter.  That is the only version of the unified claim worth reporting for it.
say "== arm B (unified) =="
if [ -s ckpts/train/lora_verifier_unified_s0/transfer_dump_pathvqa_open_lingshu7b.json ]; then
  python3 src/cascade_methods/unified_pipeline.py --analyse --tag unified_s0 \
    --open_dump_dir ckpts/train/lora_verifier_unified_s0 >>"$L" 2>&1
  say "   analyse (ONE adapter on all 8 cells) rc=$?"
else
  python3 src/cascade_methods/unified_pipeline.py --analyse --tag unified_s0 >>"$L" 2>&1
  say "   analyse (open branch = incumbent dumps; arm B's own open scores not on disk yet) rc=$?"
fi

for TAG in optiononly_s0 unified_s0; do
  for S in floors 2x2 repaired textleak fusion; do
    python3 "src/cascade_methods/unified_pipeline_${S}.py" --tag "$TAG" >>"$L" 2>&1
    say "   $S $TAG rc=$?"
  done
done

say "== open-half interference =="
python3 src/cascade_methods/unified_pipeline_openhalf.py --tags unified_s0,optiononly_s0 >>"$L" 2>&1
say "   openhalf rc=$?"

say "== finalize =="
python3 src/cascade_methods/unified_pipeline_finalize.py >>"$L" 2>&1
say "   finalize rc=$?"
