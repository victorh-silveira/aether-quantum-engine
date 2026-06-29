#!/usr/bin/env bash
set -euo pipefail

apply_overcommit() {
  if sysctl -n vm.overcommit_memory 2>/dev/null | grep -q '^1$'; then
    return 0
  fi
  if sysctl -w vm.overcommit_memory=1 2>/dev/null; then
    echo "host-prereq: vm.overcommit_memory=1 aplicado"
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    if sudo -n sysctl -w vm.overcommit_memory=1 2>/dev/null; then
      echo "host-prereq: vm.overcommit_memory=1 aplicado via sudo"
      return 0
    fi
  fi
  echo "host-prereq: aviso - nao foi possivel definir vm.overcommit_memory=1 (Redis pode alertar)"
  return 0
}

persist_overcommit() {
  local conf="/etc/sysctl.d/99-aether-redis.conf"
  if [ ! -w /etc/sysctl.d ] 2>/dev/null; then
    return 0
  fi
  if [ -f "$conf" ] && grep -q 'vm.overcommit_memory=1' "$conf" 2>/dev/null; then
    return 0
  fi
  if echo 'vm.overcommit_memory=1' >"$conf" 2>/dev/null; then
    sysctl --system >/dev/null 2>&1 || true
    echo "host-prereq: persistido em $conf"
  fi
}

main() {
  if [ -f /proc/version ] && grep -qi microsoft /proc/version; then
    apply_overcommit
    persist_overcommit
  else
    apply_overcommit || true
  fi
}

main "$@"
