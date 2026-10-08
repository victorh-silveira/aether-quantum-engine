"""Entrada do indicador publico de mercados sinteticos."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from indicator.service import run_indicator


def main() -> None:
    """Carrega configuracao do indicador e inicia observacao."""
    root = Path(__file__).resolve().parent.parent
    config = json.loads((root / "config" / "settings.json").read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    dsn = os.environ.get("AETHER_TIMESCALE_DSN", config["timescale_dsn"])
    asyncio.run(run_indicator(dsn, root / config["models_dir"], config["public_ws_url"], config["metrics_port"]))


if __name__ == "__main__":
    main()
