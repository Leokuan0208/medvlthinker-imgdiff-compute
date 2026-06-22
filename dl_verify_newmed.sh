#!/bin/bash
export HF_HOME=/data/dan/hf_cache HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd ~/medvlthinker-imgdiff-compute
for m in ddvd233/QoQ-Med-VL-7B manglu3935/Chiron-o1-2B manglu3935/Chiron-o1-8B ddvd233/QoQ-Med-VL-32B; do
  echo "=== downloading $m $(date +%H:%M) ==="
  huggingface-cli download "$m" --max-workers 4 >> logs/dl_newmed.log 2>&1
  for pass in 1 2; do
    bad=$(python3 - "$m" <<'PY'
import sys,glob,os
from safetensors import safe_open
mid=sys.argv[1].replace('/','--'); dd=glob.glob(f"/data/dan/hf_cache/hub/models--{mid}/snapshots/*/")
n=0
if dd:
  for f in sorted(glob.glob(dd[0]+"*.safetensors")):
    try:
      with safe_open(f,framework="pt") as h: list(h.keys())
    except Exception:
      os.remove(os.path.realpath(f)); n+=1
print(n)
PY
)
    echo "  $m pass$pass: $bad corrupt shard(s) removed"
    [ "$bad" -eq 0 ] && break
    huggingface-cli download "$m" --max-workers 4 >> logs/dl_newmed.log 2>&1
  done
done
echo "=== ALL_NEWMED_DOWNLOADED $(date +%H:%M) ==="
