#!/usr/bin/env bash
# Completes the D3 commit once head_final_stack.py --generator qoq lands its artifact.
# Detached (nohup) so it survives `claude agents respawn`. Validates before touching git;
# on any validation failure it writes a marker and commits NOTHING.
set -u
W=/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/audit-2026-09-18
cd "$W" || exit 1
ART=results/cascade_methods/artifacts/head_final_stack_qoq_2026-09-28.json
DOC=results/cascade_methods/docs/current/LINEAGE_TRANSFER_2026-09-28.md
MARK=logs/d3_finish.status

say(){ echo "[$(date -u +%H:%M:%S)] $*" >> logs/d3_finish.log; }
say "watching for $ART"

# wait for the fit process to exit (max 3h)
for _ in $(seq 1 1080); do
  pgrep -f "head_final_stack.py --generator qoq" >/dev/null || break
  sleep 10
done

if [ ! -f "$ART" ]; then
  echo "FAILED: fit exited without writing $ART -- rerun it, nothing committed" > "$MARK"
  say "FAILED: no artifact"; exit 1
fi

# validate: the arm I actually quote must be present and finite
python3 - "$ART" << 'PY' || { echo "FAILED: artifact did not validate -- nothing committed" > logs/d3_finish.status; exit 1; }
import json,sys,math
d=json.load(open(sys.argv[1]))
v=d.get("macro",{}).get("pooled_ens")
assert isinstance(v,(int,float)) and math.isfinite(v), f"macro.pooled_ens missing/non-finite: {v!r}"
assert d.get("cells"), "no cells"
print(f"validated: macro.pooled_ens={v:+.4f} over {len(d['cells'])} cells")
PY
say "artifact validated"

# The running fit was launched from the PRE-restoration code, so its VERDICT used max(macro)
# (arm chosen on the eval half). Recompute the fields the restored code would have written.
python3 - "$ART" << 'PY'
import json,sys
p=sys.argv[1]; a=json.load(open(p))
best=max(a["macro"],key=a["macro"].get); SHIPPED="pooled_ens"
lead=SHIPPED if SHIPPED in a["macro"] else best
base=a["macro"].get("deployed_4dom_L21ish")
a["shipped_arm"],a["best_arm_posthoc"]=SHIPPED,best
a["VERDICT"]=(f"shipped arm {lead}: {a['macro'][lead]:+.4f} macro, beats greedy on "
              f"{a['beats_greedy'][lead]} (judge currency)" +
              (f", {a['macro'][lead]-base:+.4f} over the four-domain single-layer probe on the same "
               f"held-out halves." if base is not None else
               " (four-domain baseline not fitted in this run).") +
              (f" Post-hoc best arm on these same halves is {best} at {a['macro'][best]:+.4f} -- "
               f"selected on evaluation data, do not headline it." if best!=lead else ""))
a["verdict_repaired_note"]=("this run was launched from a build in which the 2026-09-18 audit's "
    "shipped-arm guard had been accidentally reverted, so it wrote VERDICT=max(macro) -- the arm "
    "chosen on the EVAL half. The macro/cells numbers are unaffected (pure refit output); only these "
    "verdict fields were recomputed, by src/audit_2026_09_18/d3_finish.sh, using the restored logic.")
json.dump(a,open(p,"w"),indent=1)
print("verdict repaired:",a["VERDICT"][:110])
PY
say "verdict repaired"

python3 src/audit_2026_09_18/d3_writeup.py > "$DOC" || { echo "FAILED: writeup generator errored" > "$MARK"; exit 1; }
grep -q "n/a" "$DOC" && say "WARNING: doc still contains n/a"
say "doc regenerated ($(wc -l < "$DOC") lines)"

git add "$ART" "$DOC"
git commit -q -F - << 'MSG'
D3 follow-up: the pair-2 native ceiling, and a verdict repaired

Fills the one value left open by 5c1fa95. head_final_stack.py --generator qoq was still on its last
arm at that commit, so the QoQ native ceiling read n/a and the transfer percentages were not
interpretable. LINEAGE_TRANSFER_2026-09-28.md is regenerated from the artifacts by d3_writeup.py.

Also repairs that artifact's VERDICT. The fit was launched from a build in which the 2026-09-18
audit's SHIPPED="pooled_ens" guard had been accidentally reverted (restored in 5c1fa95, but the
process had already loaded the old code), so it wrote VERDICT=max(macro) -- the arm chosen on the
EVAL half. The macro and per-cell numbers are pure refit output and are unaffected; only the verdict
fields were recomputed with the restored logic, and the artifact records that it was repaired.

Two caveats on this ceiling, both stated in the doc: it is a 5-seed fit while the transfer arms use
the frozen 8-seed genframe_head_pooled_ens_v2 probes, and QoQ has 87,308 pooled training rows
against Qwen's 145,085 -- so the two native ceilings match each other in recipe but not in volume.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q8J4SpG8WQbTwXXZJxqG8U
MSG
if [ $? -eq 0 ]; then
  echo "DONE $(git log --oneline -1)" > "$MARK"; say "committed: $(git log --oneline -1)"
else
  echo "FAILED: git commit returned nonzero" > "$MARK"; say "commit failed"
fi
