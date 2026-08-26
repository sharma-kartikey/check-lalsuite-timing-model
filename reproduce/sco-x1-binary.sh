#!/usr/bin/env bash
# Compare the LAL binary timing model with PINT's BT model for one circular
# Sco X-1 example. The period and tasc use the PEGS IV central values from
# Whelan et al. (2023), arXiv:2302.10338. We use asini=1.44 light-seconds as
# a representative value. For e=0 and argp=0, tp is equivalent to tasc; the
# value below propagates tasc=1078153676 by 4274 orbits to the O4 epoch.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}

"$PYTHON" "$ROOT/scripts/checkBarycentering.py" \
  --output-dir "$ROOT/results/sco-x1-binary" \
  --gps-start 1368921618 \
  --duration 31536000 \
  --cadence 3600 \
  --detector L1 \
  --frequency 100 \
  --lal-ephemeris DE405 \
  --pint-ephemeris DE405 \
  --lal-einstein tdb \
  --pint-binary-model BT \
  --avoid-sun \
  --alpha 4.275697929502770 \
  --delta -0.272974440111460 \
  --asini 1.44 \
  --period 68023.91 \
  --tp 1368887867.34 \
  --ecc 0 \
  --argp 0
