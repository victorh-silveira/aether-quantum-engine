"""Treino offline do indicador para o catalogo sintetico atual."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from indicator.deriv import PublicDeriv
from indicator.model import load
from indicator.storage import SignalStore
from indicator.training import train_all


async def train() -> None:
    """Treina modelos por ativo e periodo usando somente OHLC publico."""
    root = Path(__file__).resolve().parent.parent
    config = json.loads((root / "config" / "settings.json").read_text(encoding="utf-8"))
    if config.get("catalog_scope") != "volatility_indices":
        raise ValueError("catalog_scope deve ser volatility_indices")
    source = PublicDeriv(config["public_ws_url"])
    models = root / config["models_dir"]
    try:
        result = await train_all(source, models)
        versions = {}
        for key in result:
            symbol, period_text = key.rsplit(":", 1)
            period = int(period_text)
            model = load(models, symbol, period, allow_experimental=True)
            if model is not None:
                versions[(symbol, period)] = model.version
        store = SignalStore(os.environ.get("AETHER_TIMESCALE_DSN", config["timescale_dsn"]))
        try:
            await store.start()
            await store.replace_models(versions)
        finally:
            await store.close()
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        await source.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    asyncio.run(train())
