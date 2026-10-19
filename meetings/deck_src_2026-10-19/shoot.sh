#!/usr/bin/env bash
# shoot.sh <deck.html> <outdir> [slide numbers...] — headless screenshots with locally unpacked libs + fonts
T=/home/jamesyang/.claude/jobs/61c9eff9/tmp
export LD_LIBRARY_PATH=$T/libs/root/usr/lib/x86_64-linux-gnu:$T/libs/root/lib/x86_64-linux-gnu
export FONTCONFIG_FILE=$T/libs/fonts.conf
export PLAYWRIGHT_BROWSERS_PATH=$T/pwbrowsers
exec $T/pwenv/bin/python $T/deck/${SHOOT_PY:-shoot.py} "$@"
