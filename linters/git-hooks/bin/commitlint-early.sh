#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MSG="${ROOT}/.git/COMMIT_EDITMSG"
if [ ! -f "${MSG}" ] || [ ! -s "${MSG}" ]; then
  echo "[commitlint] skip: mensagem ainda nao gravada"
  exit 0
fi
cd "${ROOT}"
EDIT_PATH="${MSG}"
if command -v wslpath >/dev/null 2>&1; then
  case "${MSG}" in
    /mnt/*)
      EDIT_PATH="$(wslpath -w "${MSG}")"
      ;;
  esac
fi
npx --yes -p @commitlint/cli -p @commitlint/config-conventional commitlint --config linters/commitlint.config.mjs --edit "${EDIT_PATH}"
