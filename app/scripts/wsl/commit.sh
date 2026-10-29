#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Aether Quantum Engine - Script de Commit Padronizado (Conventional Commits PT-BR)
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT_DIR}"

VALID_TYPES=("build" "chore" "ci" "docs" "feat" "fix" "perf" "qa" "refactor" "revert" "style" "test")
VALID_SCOPES=("all" "api" "app" "config" "deps" "domain" "engine" "infra" "llm" "orchestrator" "pres" "release" "repo" "risk" "scripts" "test" "tools" "ws")

FORBIDDEN_ENGLISH_WORDS=(
  "add" "adding" "added"
  "update" "updating" "updated"
  "remove" "removing" "removed"
  "refine" "refining" "refined"
  "tune" "tuning" "tuned"
  "veto" "vetoing" "vetoed"
  "tighten" "tightening" "tightened"
  "implement" "implementing" "implemented"
  "align" "aligning" "aligned"
  "fix" "fixing" "fixed"
  "test" "testing" "tested"
  "clean" "cleaning" "cleaned"
  "allow" "allowing" "allowed"
  "change" "changing" "changed"
)

show_help() {
  cat <<'EOF'
Uso do commit padronizado Aether:
  bash app/scripts/wsl/commit.sh <tipo> <escopo> "<assunto em PT-BR>" "<corpo descritivo>" [opcoes git...]
  bash app/scripts/wsl/commit.sh -t <tipo> -s <escopo> -m "<assunto>" -b "<corpo>" [-a] [--dry-run]
  bash app/scripts/wsl/commit.sh -F <arquivo_com_mensagem> [-a] [--dry-run]

Tipos permitidos (type-enum):
  build, chore, ci, docs, feat, fix, perf, qa, refactor, revert, style, test

Escopos permitidos (scope-enum):
  all, api, app, config, deps, domain, engine, infra, llm,
  orchestrator, pres, release, repo, risk, scripts, test, tools, ws

Regras obrigatorias tratadas automaticamente pelo script:
  - Assunto obrigatoriamente em Portugues (PT-BR) sem verbos/palavras em ingles.
  - Header (tipo(escopo): assunto) com no maximo 100 caracteres.
  - Linha em branco obrigatoria entre assunto e corpo.
  - Corpo da mensagem OBRIGATORIO.
  - Quebra de linha automatica no corpo para no maximo 80 caracteres (body-max-line-length).
EOF
}

check_in_array() {
  local needle="$1"
  shift
  for item in "$@"; do
    if [ "$item" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

validate_subject_pt_br() {
  local subj="$1"
  for word in "${FORBIDDEN_ENGLISH_WORDS[@]}"; do
    if echo "$subj" | grep -iqE "\b${word}\b"; then
      echo "ERRO: O assunto contem termo em ingles proibido pelo commitlint: '${word}'" >&2
      echo "Assunto avaliado: '${subj}'" >&2
      echo "Use termos em Portugues (ex: adicionar, atualizar, remover, corrigir, ajustar, alinhar)." >&2
      return 1
    fi
  done
  return 0
}

wrap_body_text() {
  local raw="$1"
  if command -v fmt >/dev/null 2>&1; then
    echo "$raw" | fmt -s -w 80
  else
    echo "$raw" | fold -s -w 80
  fi
}

TIPO=""
ESCOPO=""
ASSUNTO=""
CORPO=""
STAGE_ALL=0
DRY_RUN=0
MSG_FILE=""
EXTRA_GIT_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -a|--all)
      STAGE_ALL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -t|--type)
      TIPO="$2"
      shift 2
      ;;
    -s|--scope)
      ESCOPO="$2"
      shift 2
      ;;
    -m|--message)
      ASSUNTO="$2"
      shift 2
      ;;
    -b|--body)
      CORPO="$2"
      shift 2
      ;;
    -F|--file)
      MSG_FILE="$2"
      shift 2
      ;;
    *)
      if [ -z "$TIPO" ]; then
        TIPO="$1"
      elif [ -z "$ESCOPO" ]; then
        ESCOPO="$1"
      elif [ -z "$ASSUNTO" ]; then
        ASSUNTO="$1"
      elif [ -z "$CORPO" ]; then
        CORPO="$1"
      else
        EXTRA_GIT_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

TMP_MSG="$(mktemp /tmp/aether-commit-msg.XXXXXX)"
cleanup() {
  rm -f "${TMP_MSG}"
}
trap cleanup EXIT INT TERM

if [ -n "$MSG_FILE" ]; then
  if [ ! -f "$MSG_FILE" ]; then
    echo "ERRO: Arquivo de mensagem nao encontrado: '$MSG_FILE'" >&2
    exit 1
  fi
  HEADER="$(head -n 1 "$MSG_FILE")"
  BODY_CONTENT="$(tail -n +2 "$MSG_FILE" | sed '/^[[:space:]]*$/d')"
  WRAPPED_BODY="$(wrap_body_text "$BODY_CONTENT")"
  printf "%s\n\n%s\n" "$HEADER" "$WRAPPED_BODY" > "$TMP_MSG"
else
  if [ -z "$TIPO" ] || [ -z "$ESCOPO" ] || [ -z "$ASSUNTO" ] || [ -z "$CORPO" ]; then
    echo "ERRO: Parametros incompletos para montar o commit convencional." >&2
    echo "E necessario informar: tipo, escopo, assunto e corpo descritivo." >&2
    echo "" >&2
    show_help
    exit 1
  fi

  TIPO="$(echo "$TIPO" | tr '[:upper:]' '[:lower:]')"
  ESCOPO="$(echo "$ESCOPO" | tr '[:upper:]' '[:lower:]')"

  if ! check_in_array "$TIPO" "${VALID_TYPES[@]}"; then
    echo "ERRO: Tipo '$TIPO' invalido. Tipos aceitos: ${VALID_TYPES[*]}" >&2
    exit 1
  fi

  if ! check_in_array "$ESCOPO" "${VALID_SCOPES[@]}"; then
    echo "ERRO: Escopo '$ESCOPO' invalido. Escopos aceitos: ${VALID_SCOPES[*]}" >&2
    exit 1
  fi

  HEADER="${TIPO}(${ESCOPO}): ${ASSUNTO}"
  HEADER_LEN="${#HEADER}"
  if [ "$HEADER_LEN" -gt 100 ]; then
    echo "ERRO: O header possui ${HEADER_LEN} caracteres (limite maximo: 100)." >&2
    echo "Header: ${HEADER}" >&2
    exit 1
  fi

  if ! validate_subject_pt_br "$ASSUNTO"; then
    exit 1
  fi

  CORPO_TRIMMED="$(echo "$CORPO" | tr -d '[:space:]')"
  if [ "${#CORPO_TRIMMED}" -lt 5 ]; then
    echo "ERRO: O corpo do commit e obrigatorio e deve conter uma descricao detalhada." >&2
    exit 1
  fi

  WRAPPED_BODY="$(wrap_body_text "$CORPO")"
  printf "%s\n\n%s\n" "$HEADER" "$WRAPPED_BODY" > "$TMP_MSG"
fi

# Pre-popula .git/COMMIT_EDITMSG para validacao antecipada no commitlint-early
if [ -d ".git" ]; then
  cp "$TMP_MSG" ".git/COMMIT_EDITMSG"
fi

echo "========================================================================"
echo "Mensagem de commit formatada e validada:"
echo "------------------------------------------------------------------------"
cat "$TMP_MSG"
echo "========================================================================"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Modo --dry-run: commit nao executado no git."
  exit 0
fi

if [ "$STAGE_ALL" -eq 1 ]; then
  git add -A
fi

if git diff --cached --quiet; then
  echo "AVISO: Nenhuma alteracao no stage para commitar (git diff --cached esta vazio)." >&2
  echo "Use 'git add <arquivos>' ou a flag '-a' para adicionar alteracoes antes de commitar." >&2
  exit 1
fi

git commit -F "$TMP_MSG" "${EXTRA_GIT_ARGS[@]}"
echo "Commit realizado com sucesso!"
