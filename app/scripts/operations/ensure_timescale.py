"""Verifica se TimescaleDB esta acessivel via TCP (porta mapeada do WSL) e tenta iniciar via WSL se necessario."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "infra" / "docker"
TS_HOST = "127.0.0.1"
TS_PORT = 5432


def _port_open(host: str = TS_HOST, port: int = TS_PORT, timeout: float = 3.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _wsl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["wsl"] + list(args), capture_output=True, check=False)


def main() -> int:
    if _port_open():
        print("[AETHER] TimescaleDB ja esta acessivel em localhost:5432.")
        return 0
    print("[AETHER] TimescaleDB nao respondeu em localhost:5432. Tentando iniciar via WSL...")
    r = _wsl(
        "docker", "compose",
        "-f", str(DOCKER_DIR / "docker-compose.yml").replace("C:", "/mnt/c").replace("\\", "/"),
        "--project-directory", str(DOCKER_DIR).replace("C:", "/mnt/c").replace("\\", "/"),
        "--env-file", str(REPO_ROOT / ".env").replace("C:", "/mnt/c").replace("\\", "/"),
        "up", "-d", "timescaledb",
    )
    if r.returncode != 0:
        stderr = (r.stderr or b"").decode(errors="replace").strip()
        print(f"[AVISO] Docker Compose via WSL falhou: {stderr or 'desconhecido'}")
        return 1
    print("[AETHER] Aguardando TimescaleDB ficar acessivel...")
    for _i in range(30):
        if _port_open():
            print("[AETHER] TimescaleDB pronto (porta 5432).")
            return 0
        time.sleep(2)
    print("[AVISO] Timeout ao aguardar TimescaleDB (60s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
