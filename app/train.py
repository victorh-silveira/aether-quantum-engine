"""Treino offline do indicador para o catalogo sintetico atual."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from indicator.deriv import PublicDeriv
from indicator.training import train_all


async def train() -> None:
    """Treina modelos por ativo e periodo usando somente OHLC publico."""
    root = Path(__file__).resolve().parent.parent
    config = json.loads((root / "config" / "settings.json").read_text(encoding="utf-8"))
    source = PublicDeriv(config["public_ws_url"])
    try:
        result = await train_all(source, root / config["models_dir"])
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        await source.close()


if __name__ == "__main__":
    asyncio.run(train())
