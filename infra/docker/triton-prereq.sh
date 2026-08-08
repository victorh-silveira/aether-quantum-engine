#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
REPO="${ROOT}/triton-models"
SSOT_SYMBOL="R_10"
source "${ROOT}/docker-ui.sh"

if [ ! -f "${REPO_ROOT}/infra/docker/docker-compose.yml" ]; then
  echo "triton-prereq: execute a partir da raiz do repositorio" >&2
  exit 1
fi

model_pt_valid() {
  local file="$1"
  [ -f "$file" ] || return 1
  local magic
  magic="$(head -c 2 "$file" 2>/dev/null | od -An -tx1 | tr -d ' \n' || true)"
  [ "$magic" = "504b" ] || return 1
  unzip -tq "$file" >/dev/null 2>&1
}

ensure_ssot_placeholder() {
  local symbol_dir="${REPO}/${SSOT_SYMBOL}"
  local version_dir="${symbol_dir}/1"
  mkdir -p "$version_dir"
  if [ ! -f "${version_dir}/.gitkeep" ] && [ ! -f "${version_dir}/model.pt" ]; then
    : > "${version_dir}/.gitkeep"
  fi
}

prune_symbol_without_model() {
  local symbol_dir="$1"
  local symbol
  symbol="$(basename "$symbol_dir")"
  local version_dir="${symbol_dir}/1"
  if [ -d "$version_dir" ] && [ ! -f "${version_dir}/model.pt" ]; then
    if [ "$symbol" != "$SSOT_SYMBOL" ]; then
      rm -rf "$symbol_dir"
      docker_ui_info "removido layout legado ${symbol} (SSOT=${SSOT_SYMBOL})"
      return 0
    fi
    rm -f "${symbol_dir}/config.pbtxt"
    find "$version_dir" -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true
    if [ ! -f "${version_dir}/.gitkeep" ]; then
      : > "${version_dir}/.gitkeep"
    fi
  fi
}

main() {
  if [ ! -d "$REPO" ]; then
    mkdir -p "$REPO"
    docker_ui_info "diretorio triton-models criado"
  fi
  ensure_ssot_placeholder
  shopt -s nullglob
  for model_pt in "$REPO"/*/*/model.pt; do
    if model_pt_valid "$model_pt"; then
      continue
    fi
    docker_ui_warn "removendo model.pt invalido: $model_pt"
    rm -f "$model_pt"
    prune_symbol_without_model "$(dirname "$(dirname "$model_pt")")"
  done
  for symbol_dir in "$REPO"/*/; do
    [ -d "$symbol_dir" ] || continue
    prune_symbol_without_model "$symbol_dir"
  done
  ensure_ssot_placeholder
}

main "$@"
