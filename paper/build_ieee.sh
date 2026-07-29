#!/usr/bin/env bash
# build_ieee.sh — compile an IEEEtran (dual-column) LaTeX paper to PDF.
#
# Engine: tectonic (single-binary LaTeX; auto-downloads IEEEtran + packages on
# first run, then caches them under ~/.cache/Tectonic). Installed locally at
# <repo>/tools/tectonic. That binary needs libgraphite2.so.3, which is not in
# the system lib path here, so we prepend a minimal lib dir that symlinks it.
#
# Usage:
#   paper/build_ieee.sh <paper.tex> [output_dir]
#
# Produces <basename>.pdf next to the .tex (or in output_dir if given).
# Run from anywhere; paths are resolved relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TECTONIC="$REPO_ROOT/tools/tectonic"
TEC_LIB="$REPO_ROOT/tools/lib"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <paper.tex> [output_dir]" >&2
  exit 2
fi

TEX="$1"
if [[ ! -f "$TEX" ]]; then
  echo "error: input .tex not found: $TEX" >&2
  exit 1
fi
TEX_DIR="$(cd "$(dirname "$TEX")" && pwd)"
TEX_BASE="$(basename "$TEX")"
OUT_DIR="${2:-$TEX_DIR}"
mkdir -p "$OUT_DIR"

# Prefer the local tectonic; fall back to one on PATH.
if [[ -x "$TECTONIC" ]]; then
  export LD_LIBRARY_PATH="$TEC_LIB:${LD_LIBRARY_PATH:-}"
  ENGINE="$TECTONIC"
elif command -v tectonic >/dev/null 2>&1; then
  ENGINE="$(command -v tectonic)"
else
  echo "error: no tectonic binary found (looked at $TECTONIC and PATH)" >&2
  exit 1
fi

echo ">> engine: $ENGINE"
echo ">> input:  $TEX_DIR/$TEX_BASE"
echo ">> outdir: $OUT_DIR"

# --keep-logs helps debugging; --outdir controls where the PDF lands.
# Compiling from TEX_DIR so a local IEEEtran.cls / figs are found.
( cd "$TEX_DIR" && "$ENGINE" --keep-logs --outdir "$OUT_DIR" "$TEX_BASE" )

PDF="$OUT_DIR/${TEX_BASE%.tex}.pdf"
if [[ -f "$PDF" ]]; then
  echo ">> OK: built $PDF"
else
  echo ">> FAILED: expected $PDF not produced" >&2
  exit 1
fi
