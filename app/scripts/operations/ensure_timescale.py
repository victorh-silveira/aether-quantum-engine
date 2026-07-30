"""Tenta iniciar TimescaleDB via Docker Compose e aguarda healthcheck."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "infra" / "docker"


def _find_docker() -> str:
    for candidates in ("docker.exe", "docker"):
        path = shutil.which(candidates)
        if path:
            return path
    return "docker"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, shell=True)


def main() -> int:
    docker = _find_docker()
    print("[AETHER] Subindo TimescaleDB via Docker Compose...")
    r = _run(
        docker, "compose",
        "-f", str(DOCKER_DIR / "docker-compose.yml"),
        "--project-directory", str(DOCKER_DIR),
        "--env-file", str(REPO_ROOT / ".env"),
        "up", "-d", "timescaledb",
    )
    if r.returncode != 0:
        stderr = (r.stderr or b"").decode(errors="replace").strip()
        print(f"[AVISO] Docker Compose falhou: {stderr or 'desconhecido'}")
        return 1
    print("[AETHER] Aguardando TimescaleDB ficar saudavel...")
    for i in range(30):
        hr = _run(docker, "exec", "aether-timescaledb", "pg_isready", "-U", "aether")
        if hr.returncode == 0:
            print("[AETHER] TimescaleDB pronto.")
            return 0
        time.sleep(2)
    print("[AVISO] Timeout ao aguardar TimescaleDB (60s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
