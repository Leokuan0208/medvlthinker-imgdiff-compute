#!/bin/bash
# Wave-driven campaign: keep GPU0, GPU1 and the CPU busy until there is genuinely nothing left.
#
# A FIXED queue cannot do this.  The first attempt used three static lists and GPU 0 drained its 12
# jobs and idled at 16 MiB while GPU 1 still had hours of work -- because the set of runnable jobs
# GROWS as earlier ones land (each new judge file makes an extraction runnable, each new feature
# cache makes a measurement runnable).  So after every wave we re-plan from what is on disk.
#
# plan_next_wave.py emits only jobs whose inputs exist and whose outputs do not, and exits 3 when it
# has nothing to emit -- that is the campaign's termination condition.  The supervisor skips jobs
# already ok in its journal, so a restart resumes instead of repeating.
set -u
cd ~/medvlthinker-imgdiff-compute
WAVE=${1:-2}
MAXWAVE=40
while [ "$WAVE" -le "$MAXWAVE" ]; do
  echo "[$(date -u +%F\ %H:%M:%S)] === planning wave $WAVE ==="
  python3 src/reporting/plan_next_wave.py --wave "$WAVE" > "logs/plan_wave$WAVE.log" 2>&1
  rc=$?
  tail -3 "logs/plan_wave$WAVE.log"
  if [ "$rc" -eq 3 ]; then echo "[$(date -u +%H:%M:%S)] nothing left to run -- CAMPAIGN COMPLETE"; break; fi
  declare -A PID=()
  for lane in gpu0 gpu1 cpu; do
    q="runners/auto_${lane}_wave${WAVE}.json"
    [ -f "$q" ] || continue
    nohup python3 -u src/reporting/supervisor.py --queue "$q" --retries 2 \
        > "logs/auto_${lane}_wave${WAVE}.log" 2>&1 &
    PID[$lane]=$!
    echo "  lane $lane wave $WAVE pid ${PID[$lane]}"
  done
  # watchdog: restart a lane that died before its queue completed
  while true; do
    alive=0
    for lane in "${!PID[@]}"; do
      if kill -0 "${PID[$lane]}" 2>/dev/null; then alive=$((alive+1)); continue; fi
      log="logs/auto_${lane}_wave${WAVE}.log"
      grep -q "QUEUE COMPLETE" "$log" 2>/dev/null && continue
      echo "[$(date -u +%H:%M:%S)] lane $lane died mid-wave -- restarting"
      nohup python3 -u src/reporting/supervisor.py \
          --queue "runners/auto_${lane}_wave${WAVE}.json" --retries 2 >> "$log" 2>&1 &
      PID[$lane]=$!; alive=$((alive+1))
    done
    [ "$alive" -eq 0 ] && break
    sleep 120
  done
  echo "[$(date -u +%F\ %H:%M:%S)] wave $WAVE done"
  WAVE=$((WAVE+1))
done
echo "[$(date -u +%F\ %H:%M:%S)] DRIVER EXIT at wave $WAVE"
