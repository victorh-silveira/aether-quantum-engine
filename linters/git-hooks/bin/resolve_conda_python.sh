#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${AETHER_CONDA_ENV:-deriv-api}"

if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "${CONDA_PREFIX}")" = "${CONDA_ENV}" ]; then
  if [ -x "${CONDA_PREFIX}/python.exe" ]; then
    echo "${CONDA_PREFIX}/python.exe"
    exit 0
  fi
  if [ -x "${CONDA_PREFIX}/bin/python" ]; then
    echo "${CONDA_PREFIX}/bin/python"
    exit 0
  fi
fi

home="${HOME:-/home/${USER}}"
for root in \
  "${home}/anaconda3" \
  "${home}/miniconda3" \
  "/mnt/c/Users/${USER}/anaconda3" \
  "/mnt/c/Users/${USER}/miniconda3" \
  "/c/Users/${USER}/anaconda3" \
  "/c/Users/${USER}/miniconda3"; do
  for py in \
    "${root}/envs/${CONDA_ENV}/python.exe" \
    "${root}/envs/${CONDA_ENV}/bin/python"; do
    if [ -x "$py" ]; then
      echo "$py"
      exit 0
    fi
  done
done

exit 1
