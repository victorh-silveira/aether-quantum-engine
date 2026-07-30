"""Verifica se TimescaleDB esta acessivel e populado, sementeia M5 via Deriv se vazio."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKER_DIR = _REPO_ROOT / "infra" / "docker"
_TS_HOST = "127.0.0.1"
_TS_PORT = 5432
_DEFAULT_DSN = "postgresql://aether:aether@localhost:5432/aether"
_MIN_BARS = 96

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AETHER")


def _port_open(timeout: float = 3.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((_TS_HOST, _TS_PORT)) == 0


def _wsl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["wsl"] + list(args), capture_output=True, check=False)


def _settings_dsn() -> str:
    settings_path = _REPO_ROOT / "config" / "settings.json"
    if settings_path.is_file():
        try:
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
            infra = raw.get("infra", {})
            chunk = infra.get("timescale", {}) if isinstance(infra, dict) else {}
            if isinstance(chunk, dict) and chunk.get("dsn"):
                return str(chunk["dsn"])
        except (OSError, json.JSONDecodeError):
            pass
    return os.getenv("AETHER_TIMESCALE_DSN", _DEFAULT_DSN)


async def _data_ok(dsn: str, symbols: list[str]) -> bool:
    try:
        import asyncpg  # noqa: PLC0415
    except ImportError:
        logger.warning("[AETHER] asyncpg nao disponivel - pulando verificacao de dados.")
        return True
    try:
        conn = await asyncpg.connect(dsn, timeout=5.0)
    except Exception as exc:
        logger.warning("[AETHER] Conexao TimescaleDB falhou: %s", exc)
        return False
    try:
        rows = await conn.fetch(
            """
            SELECT symbol, granularity, COUNT(*)::int AS total
            FROM ohlc_bars
            WHERE symbol = ANY($1::text[])
            GROUP BY symbol, granularity
            """,
            symbols,
        )
        counts: dict[tuple[str, int], int] = {(r["symbol"], r["granularity"]): r["total"] for r in (rows or [])}
        for sym in symbols:
            for gran in (300, 600, 900):
                have = counts.get((sym, gran), 0)
                if have < _MIN_BARS:
                    logger.info(
                        "[AETHER] TimescaleDB | %s gran=%ds tem %d barras (min=%d)",
                        sym,
                        gran,
                        have,
                        _MIN_BARS,
                    )
                    return False
        return True
    finally:
        await conn.close()


def _seed_timescale(symbols: list[str]) -> int:
    seed_script = str(_REPO_ROOT / "app" / "scripts" / "operations" / "seed_timescale_ohlc.py")
    cmd = [sys.executable, seed_script, "--symbols"] + symbols
    logger.info("[AETHER] Sementeando TimescaleDB via Deriv API (seed_timescale_ohlc.py)...")
    r = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
    if r.returncode != 0:
        stderr = (r.stderr or b"").decode(errors="replace").strip()[:500]
        logger.warning("[AVISO] Seed TimescaleDB falhou: %s", stderr or "desconhecido")
        return 1
    logger.info("[AETHER] TimescaleDB sementeado com sucesso.")
    return 0


def _ensure_timescaledb_running() -> int:
    if _port_open():
        logger.info("[AETHER] TimescaleDB acessivel em localhost:5432.")
        return 0
    logger.info("[AETHER] TimescaleDB nao respondeu. Tentando iniciar via WSL...")
    r = _wsl(
        "docker",
        "compose",
        "-f",
        str(_DOCKER_DIR / "docker-compose.yml").replace("C:", "/mnt/c").replace("\\", "/"),
        "--project-directory",
        str(_DOCKER_DIR).replace("C:", "/mnt/c").replace("\\", "/"),
        "--env-file",
        str(_REPO_ROOT / ".env").replace("C:", "/mnt/c").replace("\\", "/"),
        "up",
        "-d",
        "timescaledb",
    )
    if r.returncode != 0:
        stderr = (r.stderr or b"").decode(errors="replace").strip()
        logger.warning("[AVISO] Docker Compose via WSL falhou: %s", stderr or "desconhecido")
        return 1
    for _i in range(30):
        if _port_open():
            logger.info("[AETHER] TimescaleDB pronto (porta 5432).")
            return 0
        time.sleep(2)
    logger.warning("[AVISO] Timeout ao aguardar TimescaleDB (60s).")
    return 1


def main() -> int:
    rc = _ensure_timescaledb_running()
    if rc != 0:
        return rc

    dsn = _settings_dsn()
    symbols = ["R_10"]
    ok = asyncio.run(_data_ok(dsn, symbols))
    if ok:
        logger.info("[AETHER] TimescaleDB | dados OHLC suficientes.")
        return 0

    logger.info("[AETHER] TimescaleDB | dados OHLC insuficientes - sementeando via Deriv...")
    return _seed_timescale(symbols)


if __name__ == "__main__":
    sys.exit(main())
