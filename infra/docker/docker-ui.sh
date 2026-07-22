#!/usr/bin/env bash

DOCKER_UI_BOLD=$'\033[1m'
DOCKER_UI_DIM=$'\033[2m'
DOCKER_UI_GREEN=$'\033[1;32m'
DOCKER_UI_YELLOW=$'\033[1;33m'
DOCKER_UI_BLUE=$'\033[1;34m'
DOCKER_UI_CYAN=$'\033[1;36m'
DOCKER_UI_RED=$'\033[1;31m'
DOCKER_UI_RESET=$'\033[0m'

docker_ui_nl() {
  printf '\n'
}

docker_ui_rule() {
  printf '%s------------------------------------------------------------------------%s\n' "${DOCKER_UI_BLUE}" "${DOCKER_UI_RESET}"
}

docker_ui_banner() {
  local title="$1"
  docker_ui_nl
  docker_ui_rule
  printf '%s  %s%s\n' "${DOCKER_UI_BOLD}" "${title}" "${DOCKER_UI_RESET}"
  docker_ui_rule
  docker_ui_nl
}

docker_ui_step() {
  local index="$1"
  local total="$2"
  local label="$3"
  docker_ui_nl
  printf '%s[%s/%s]%s %s%s%s\n' \
    "${DOCKER_UI_CYAN}" "${index}" "${total}" "${DOCKER_UI_RESET}" \
    "${DOCKER_UI_BOLD}" "${label}" "${DOCKER_UI_RESET}"
  docker_ui_nl
}

docker_ui_info() {
  printf '  %s%s%s\n' "${DOCKER_UI_DIM}" "$*" "${DOCKER_UI_RESET}"
}

docker_ui_ok() {
  printf '  %-28s %sOK%s\n' "$1" "${DOCKER_UI_GREEN}" "${DOCKER_UI_RESET}"
}

docker_ui_warn() {
  printf '  %sAVISO%s  %s\n' "${DOCKER_UI_YELLOW}" "${DOCKER_UI_RESET}" "$*"
}

docker_ui_fail() {
  printf '  %-28s %sFALHA%s  %s\n' "$1" "${DOCKER_UI_RED}" "${DOCKER_UI_RESET}" "${2:-}"
}

docker_ui_done() {
  local label="$1"
  docker_ui_nl
  docker_ui_rule
  printf '  %s%s%s\n' "${DOCKER_UI_GREEN}" "${label}" "${DOCKER_UI_RESET}"
  docker_ui_rule
  docker_ui_nl
}
