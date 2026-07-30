"""Tenta iniciar TimescaleDB via Docker Compose e aguarda healthcheck."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "infra" / "docker"
COMPOSE_FILE = DOCKER_DIR / "docker-compose.yml"
ENV_FILE = REPO_ROOT / ".env"


def main() -> int:
    print("[AETHER] Subindo TimescaleDB via Docker Compose...")
    cmd = [
        "docker", "compose",
        "-f", str(COMPOSE_FILE),
        "--project-directory", str(DOCKER_DIR),
        "--env-file", str(ENV_FILE),
        "up", "-d", "timescaledb",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        stderr = r.stderr.decode(errors="replace").strip()
        print(f"[AVISO] Docker Compose falhou: {stderr or 'desconhecido'}")
        return 1
    print("[AETHER] Aguardando TimescaleDB ficar saudavel...")
    for i in range(30):
        hr = subprocess.run(
            ["docker", "exec", "aether-timescaledb", "pg_isready", "-U", "aether"],
            capture_output=True,
        )
        if hr.returncode == 0:
            print("[AETHER] TimescaleDB pronto.")
            return 0
        time.sleep(2)
    print("[AVISO] Timeout ao aguardar TimescaleDB (60s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
