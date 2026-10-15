"""Treino offline do indicador para o catalogo sintetico atual."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from indicator.deriv import PublicDeriv
from indicator.pullback_store import PullbackStore
from indicator.pullback_training import diagnose_all


async def train() -> None:
    """Prepara dados e diagnosticos do setup fixo por ativo e periodo."""
    root = Path(__file__).resolve().parent.parent
    config = json.loads((root / "config" / "settings.json").read_text(encoding="utf-8"))
    if config.get("catalog_scope") != "volatility_indices":
        raise ValueError("catalog_scope deve ser volatility_indices")
    source = PublicDeriv(config["public_ws_url"])
    store = PullbackStore(os.environ.get("AETHER_TIMESCALE_DSN", config["timescale_dsn"]))
    variant = os.environ.get("PULLBACK_VARIANT", config.get("pullback_variant", "base"))
    try:
        await store.start()
        result = await diagnose_all(source, store, variant=variant)
        summary = {
            "pares": len(result),
            "pares_com_proxy": sum(row["estado"] == "PROXY DE VELAS" for row in result.values()),
            "pares_sem_sinal": sum(row["estado"] == "SEM SINAL" for row in result.values()),
            "setups": sum(row.get("setups", 0) for row in result.values()),
            "acertos_proxy": sum(row.get("acertos", 0) for row in result.values()),
            "erros_proxy": sum(row.get("erros", 0) for row in result.values()),
            "empates_proxy": sum(row.get("empates", 0) for row in result.values()),
            "qualificacao_broker": "nao realizada",
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        await store.close()
        await source.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    asyncio.run(train())
