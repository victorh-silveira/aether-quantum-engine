#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/victor-silveira/Desktop/GitHub/aether-quantum-engine
git add -A
git status --short
git commit -m "$(cat <<'EOF'
feat(engine): FLIP sticky, explos firme e cover soft

SCALE nao desfaz FLIP (flip_holds); explos so com Edge TCN mole
(adapt_explos_max_tcn_edge 0.05); FLIP exige n_train>=8; recovery
cover_enabled com cover_multiple 1.0 e caps L0-L3. Inclui pausa
until SSOT, META sat soft e imagens MinIO pgsty.

EOF
)"
git status
