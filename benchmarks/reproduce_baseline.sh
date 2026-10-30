#!/usr/bin/env bash
# Reproduce the official pygeoflood benchmark baseline (commit 3650c9b)
# from a clean venv on the pinned sandbox (8 vCPU / 7.77 GB — the manifest
# records actuals, do not trust this comment over the data).
#
# Usage: bash benchmarks/reproduce_baseline.sh [VENV_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${1:-.venv-bench}"

if [ ! -x "$VENV/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" -e ".[bench]"
  elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$VENV"
    "$VENV/bin/pip" install -e ".[bench]"
  else
    echo "error: need uv or python3.12 to build the benchmark venv" >&2
    exit 1
  fi
fi

# COBOL kernel rows: compile the backend when GnuCOBOL is present so a
# PYGEOFLOOD_COBOL=1 rerun of this script can produce backend-tagged rows.
# Optional: without cobc the baseline reproduces on the default (Rust) route.
if command -v cobc >/dev/null 2>&1; then
  make -C cobol lib
fi

# Headline profile: full 16.0M-pixel tile, cold + 2 warm.
"$VENV/bin/python" -m benchmarks --profile full --warm-runs 2 \
  --outdir benchmarks/results/reproduce-full

# Iteration profile: fixed 2000x2000 crop, cold + 2 warm.
"$VENV/bin/python" -m benchmarks --profile crop --warm-runs 2 \
  --outdir benchmarks/results/reproduce-crop

echo
echo "Baseline reproduced. Results:"
ls -d benchmarks/results/reproduce-*
