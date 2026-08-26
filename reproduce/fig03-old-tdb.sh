#!/usr/bin/env bash
# Generate the timing-component data used for Fig. 3: old LAL TT-TDB
# correction versus PINT TT(TAI), at the paper's three sky positions.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}

"$PYTHON" "$ROOT/scripts/checkBarycentering.py" \
  --output-dir "$ROOT/results/fig03-old-tdb" \
  --gps-start 1368921618 \
  --duration 31536000 \
  --cadence 3600 \
  --detector L1 \
  --frequency 100 \
  --lal-ephemeris DE405 \
  --pint-ephemeris DE405 \
  --lal-einstein tdb_old \
  --avoid-sun \
  --sky-position 1.1000028170845237 -0.99003449096536966 \
  --sky-position 3.9302759765568003 -0.088003185184927488 \
  --sky-position 1.8501629005379903  0.65210958983022616
