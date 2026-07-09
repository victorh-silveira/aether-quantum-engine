#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ROOT}/triton-models"

model_pt_valid() {
  local file="$1"
  [ -f "$file" ] || return 1
  local magic
  magic="$(head -c 2 "$file" 2>/dev/null | od -An -tx1 | tr -d ' \n' || true)"
  [ "$magic" = "504b" ] || return 1
  unzip -tq "$file" >/dev/null 2>&1
}

prune_symbol_without_model() {
  local symbol_dir="$1"
  local version_dir="${symbol_dir}/1"
  if [ -d "$version_dir" ] && [ ! -f "${version_dir}/model.pt" ]; then
    rm -f "${symbol_dir}/config.pbtxt"
    find "$version_dir" -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true
    echo "triton-prereq: layout vazio em $(basename "$symbol_dir") (modelo sera criado pelo make train)"
  fi
}

main() {
  if [ ! -d "$REPO" ]; then
    mkdir -p "$REPO"
    return 0
  fi
  shopt -s nullglob
  for model_pt in "$REPO"/*/*/model.pt; do
    if model_pt_valid "$model_pt"; then
      continue
    fi
    echo "triton-prereq: removendo model.pt invalido: $model_pt"
    rm -f "$model_pt"
    prune_symbol_without_model "$(dirname "$(dirname "$model_pt")")"
  done
  for symbol_dir in "$REPO"/*/; do
    [ -d "$symbol_dir" ] || continue
    prune_symbol_without_model "$symbol_dir"
  done
}

main "$@"
