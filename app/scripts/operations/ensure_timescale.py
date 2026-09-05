"""Verifica se TimescaleDB esta acessivel e populado, sementeia OHLC via Deriv se vazio."""

from __future__ import annotations

import argparse
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
_APP_ROOT = _REPO_ROOT / "app"
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from scripts.operations.timescale_seed_policy import (  # noqa: E402
    MIN_BARS_MICRO,
    min_bars_for_granularity,
)

_DOCKER_DIR = _REPO_ROOT / "infra" / "docker"
_TS_HOST = "127.0.0.1"
_TS_PORT = 5432
_DEFAULT_DSN = "postgresql://aether:aether@localhost:5432/aether"
_SEED_TIMEOUT_SECONDS = 900

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AETH.ops")


def _port_open(timeout: float = 3.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((_TS_HOST, _TS_PORT)) == 0


def _wsl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["wsl"] + list(args), capture_output=True, check=False)


def _load_settings() -> dict:
    settings_path = _REPO_ROOT / "config" / "settings.json"
    if settings_path.is_file():
        try:
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _settings_dsn(settings: dict | None = None) -> str:
    raw = settings if isinstance(settings, dict) else _load_settings()
    infra = raw.get("infra", {})
    chunk = infra.get("timescale", {}) if isinstance(infra, dict) else {}
    if isinstance(chunk, dict) and chunk.get("dsn"):
        return str(chunk["dsn"])
    return os.getenv("AETHER_TIMESCALE_DSN", _DEFAULT_DSN)


def _required_granularities(settings: dict) -> list[int]:
    data = settings.get("data_handler") if isinstance(settings.get("data_handler"), dict) else {}
    micro = int(data.get("micro_granularity", 60) or 60)
    macro = int(data.get("granularity", 300) or 300)
    ordered = [micro, macro]
    unique: list[int] = []
    for value in ordered:
        if value not in unique:
            unique.append(value)
    return unique


async def _data_ok(
    dsn: str,
    symbols: list[str],
    granularities: list[int],
    *,
    log_shortfalls: bool = True,
) -> bool:
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
            for gran in granularities:
                floor = min_bars_for_granularity(int(gran))
                have = counts.get((sym, int(gran)), 0)
                if have < floor:
                    if log_shortfalls:
                        logger.info(
                            "[AETHER] TimescaleDB | %s gran=%ds tem %d barras (min=%d)",
                            sym,
                            int(gran),
                            have,
                            floor,
                        )
                    return False
        return True
    finally:
        await conn.close()


def _seed_timescale(symbols: list[str]) -> int:
    seed_script = str(_REPO_ROOT / "app" / "scripts" / "operations" / "seed_timescale_ohlc.py")
    cmd = [sys.executable, seed_script, "--bars", str(MIN_BARS_MICRO), "--symbols"] + symbols
    logger.info(
        "[AETHER] Sementeando TimescaleDB via Deriv (timeout=%ds, M5=%d D1=365)...",
        _SEED_TIMEOUT_SECONDS,
        MIN_BARS_MICRO,
    )
    try:
        r = subprocess.run(cmd, timeout=_SEED_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        logger.warning(
            "[AVISO] Seed TimescaleDB timeout apos %ds; meta usara Deriv se preciso.",
            _SEED_TIMEOUT_SECONDS,
        )
        return 1
    if r.returncode != 0:
        logger.warning("[AVISO] Seed TimescaleDB falhou rc=%s", r.returncode)
        return 1
    logger.info("[AETHER] TimescaleDB sementeado com sucesso.")
    return 0


def _ensure_timescaledb_running(*, quiet: bool = False) -> int:
    if _port_open():
        if not quiet:
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
    parser = argparse.ArgumentParser(description="Garante TimescaleDB para treino meta")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="So valida porta/DSN; nao sementeia OHLC (meta usa --source auto)",
    )
    args = parser.parse_args()
    check_only = bool(args.check_only)
    rc = _ensure_timescaledb_running(quiet=check_only)
    if rc != 0:
        return rc

    settings = _load_settings()
    dsn = _settings_dsn(settings)
    symbols = [str(s) for s in (settings.get("symbols") or ["1HZ75V"])]
    granularities = _required_granularities(settings)
    ok = asyncio.run(_data_ok(dsn, symbols, granularities, log_shortfalls=not check_only))
    if ok:
        if check_only:
            logger.info(
                "[AETHER] Timescale check-only: ok porta=%s ohlc=meta_ready gran=%s",
                _TS_PORT,
                granularities,
            )
        else:
            logger.info("[AETHER] TimescaleDB | dados OHLC meta_ready.")
        return 0

    if check_only:
        logger.info(
            "[AETHER] Timescale check-only: ok porta=%s ohlc=smoke gran=%s → seed/Deriv antes do meta",
            _TS_PORT,
            granularities,
        )
        return 0

    logger.info("[AETHER] TimescaleDB | ohlc=smoke/curto - sementeando via Deriv...")
    return _seed_timescale(symbols)


if __name__ == "__main__":
    sys.exit(main())
