#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV="deriv-api"

if [ ! -f /proc/version ] || ! grep -qi microsoft /proc/version; then
  echo "Este script e apenas para WSL." >&2
  exit 1
fi

win_gitconfig="/mnt/c/Users/${USER}/.gitconfig"
if [ -f "$win_gitconfig" ]; then
  win_name="$(git config -f "$win_gitconfig" user.name 2>/dev/null || true)"
  win_email="$(git config -f "$win_gitconfig" user.email 2>/dev/null || true)"
  if [ -n "$win_name" ] && [ -z "$(git config --global user.name 2>/dev/null || true)" ]; then
    git config --global user.name "$win_name"
  fi
  if [ -n "$win_email" ] && [ -z "$(git config --global user.email 2>/dev/null || true)" ]; then
    git config --global user.email "$win_email"
  fi
fi

bashrc="${HOME}/.bashrc"
if grep -q 'profile.d/conda.sh' "$bashrc" 2>/dev/null; then
  sed -i '/profile\.d\/conda\.sh/,+2d' "$bashrc"
  echo "Removido bloco conda.sh invalido do Windows em ~/.bashrc"
fi

conda_exe=""
env_python=""
for root in \
  "${HOME}/anaconda3" \
  "${HOME}/miniconda3" \
  "/mnt/c/Users/${USER}/anaconda3" \
  "/mnt/c/Users/${USER}/miniconda3"; do
  candidate_exe="${root}/Scripts/conda.exe"
  candidate_py="${root}/envs/${CONDA_ENV}/python.exe"
  if [ -z "$conda_exe" ] && [ -x "$candidate_exe" ]; then
    conda_exe="$candidate_exe"
  fi
  if [ -z "$env_python" ] && [ -x "$candidate_py" ]; then
    env_python="$candidate_py"
  fi
  candidate_py_linux="${root}/envs/${CONDA_ENV}/bin/python"
  if [ -z "$env_python" ] && [ -x "$candidate_py_linux" ]; then
    env_python="$candidate_py_linux"
  fi
done

if ! grep -q "AETHER_QUANTUM_ENGINE_WSL=1" "$bashrc" 2>/dev/null; then
  {
    echo ""
    if [ -n "$conda_exe" ]; then
      echo "alias conda='${conda_exe}'"
    fi
    if [ -n "$env_python" ]; then
      echo "export AETHER_PYTHON='${env_python}'"
      echo "aether-py() { \"\${AETHER_PYTHON}\" \"\$@\"; }"
    fi
    echo "export AETHER_CONDA_ENV=\"${CONDA_ENV}\""
    echo "export AETHER_QUANTUM_ENGINE_WSL=1"
  } >> "$bashrc"
  echo "Bloco WSL do projeto adicionado em ~/.bashrc"
else
  appended=0
  if [ -n "$conda_exe" ] && ! grep -q "alias conda=" "$bashrc" 2>/dev/null; then
    echo "alias conda='${conda_exe}'" >> "$bashrc"
    appended=1
    echo "Alias conda adicionado em ~/.bashrc"
  fi
  if [ -n "$env_python" ] && ! grep -q "AETHER_PYTHON=" "$bashrc" 2>/dev/null; then
    {
      echo "export AETHER_PYTHON='${env_python}'"
      echo "aether-py() { \"\${AETHER_PYTHON}\" \"\$@\"; }"
    } >> "$bashrc"
    appended=1
    echo "Helper aether-py adicionado em ~/.bashrc"
  fi
  if [ "$appended" -eq 0 ]; then
    echo "~/.bashrc ja contem configuracao WSL do projeto"
  fi
fi

cd "$ROOT"
chmod +x linters/git-hooks/bin/resolve_conda_python.sh linters/git-hooks/bin/python
if [ -f "$ROOT/infra/docker/host-prereq.sh" ]; then
  bash "$ROOT/infra/docker/host-prereq.sh" || true
fi
make pre-commit

py="$(bash linters/git-hooks/bin/resolve_conda_python.sh)"
echo ""
echo "Verificacao:"
echo "  git user: $(git config --global user.name) <$(git config --global user.email)>"
echo "  python:   $py"
"$py" --version
"$py" -m pre_commit --version
if [ -n "$conda_exe" ]; then
  "$conda_exe" run -n "$CONDA_ENV" python --version
fi
echo ""
echo "Proximos passos:"
echo "  source ~/.bashrc"
echo "  cd ${ROOT} && make install"
echo "  make test"
echo ""
echo "No WSL use make/conda run; conda activate do Windows nao funciona no bash nativo."
echo "  conda run -n ${CONDA_ENV} python app/scripts/..."
