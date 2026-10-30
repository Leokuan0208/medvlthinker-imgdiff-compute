#!/usr/bin/env bash
# gpu_run.sh G cmd... -- take GPU G's lock, pin the job to it, run cmd. A real executable, so `timeout` can run it
# (a bash function cannot be exec'd by timeout: that bug made every `timeout ... on G ...` step exit 127, 2026-10-22).
g=$1; shift
exec flock -w 172800 /home/jamesyang/medvlthinker-imgdiff-compute/logs/.gpu$g.lock env CUDA_VISIBLE_DEVICES=$g "$@"
