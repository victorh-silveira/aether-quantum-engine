#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/linters/git-hooks"
DEST="$ROOT/.git/hooks"

if [ ! -d "$DEST" ]; then
  echo "Diretorio .git/hooks ausente. Inicialize o repositorio git primeiro." >&2
  exit 1
fi

for name in pre-commit commit-msg; do
  install -m 755 "$SRC/$name" "$DEST/$name"
done
chmod +x "$SRC/bin/python"

echo "Hooks instalados em .git/hooks (pre-commit, commit-msg)"
