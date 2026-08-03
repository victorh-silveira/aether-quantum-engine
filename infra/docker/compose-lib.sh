#!/usr/bin/env bash

compose_args() {
  local args=(-f infra/docker/docker-compose.yml)
  if [ "${DOCKER_GPU:-1}" = "1" ]; then
    args+=(-f infra/docker/docker-compose.gpu.yml)
  fi
  args+=(--project-directory infra/docker)
  if [ -f .env ]; then
    args+=(--env-file .env)
  fi
  printf '%s\n' "${args[@]}"
}

compose_profiles_csv() {
  echo "${COMPOSE_PROFILES:-${DOCKER_PROFILES:-core,gpu,ml}}" | tr -d ' '
}

profile_active() {
  local needle="$1"
  local csv
  csv="$(compose_profiles_csv)"
  case ",${csv}," in
    *",${needle},"*) return 0 ;;
    *) return 1 ;;
  esac
}
