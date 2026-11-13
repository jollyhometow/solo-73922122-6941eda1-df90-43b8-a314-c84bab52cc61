#!/usr/bin/env bash
# Entrypoint: make sure the fixed historical baseline and demo samples exist,
# then exec the given command (uvicorn by default).
set -euo pipefail

: "${DATA_DIR:=/data}"
BASELINE="${DATA_DIR}/baseline/baseline.nc"

if [[ ! -f "${BASELINE}" ]]; then
    echo "[entrypoint] generating synthetic baseline and sample targets in ${DATA_DIR}"
    python -I -m scripts.generate_samples --data-dir "${DATA_DIR}"
else
    echo "[entrypoint] reusing existing baseline ${BASELINE}"
fi

exec "$@"
