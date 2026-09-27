"""Journal local de propostas e artefatos Touch; ticks permanecem no Timescale."""

import json
from pathlib import Path

import asyncpg

from aether_paths import repo_path


def append_touch_quotes(quotes: list[dict]) -> None:
    """Persiste todas as alternativas, nao apenas operacoes vencedoras/selecionadas."""
    path = repo_path("data", "touch", "quotes.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for quote in quotes:
            handle.write(json.dumps(quote, allow_nan=False) + "\n")


def read_touch_quotes() -> list[dict]:
    """Falha explicitamente em journal ausente ou corrompido."""
    path = repo_path("data", "touch", "quotes.jsonl")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_touch_model() -> dict:
    """Carrega JSON sem desserializar codigo executavel."""
    return json.loads(repo_path("data", "touch", "model.json").read_text(encoding="utf-8"))


def save_touch_model(bundle: dict) -> Path:
    """Troca atomica: falha no treino nunca destrui o ultimo artefato."""
    path = repo_path("data", "touch", "model.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(bundle, allow_nan=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


async def read_touch_ticks(dsn: str, symbol: str, start_ms: int, end_ms: int) -> list[tuple]:
    """Busca intervalo parametrizado; duplicatas exatas sao eliminadas, conflitos rejeitados depois."""
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            "SELECT DISTINCT epoch_ms, price FROM ticks WHERE symbol=$1 AND epoch_ms BETWEEN $2 AND $3 "
            "ORDER BY epoch_ms",
            symbol,
            start_ms,
            end_ms,
        )
        return [(int(row["epoch_ms"]), float(row["price"])) for row in rows]
    finally:
        await conn.close()
