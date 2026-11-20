"""Entrada do indicador publico de mercados sinteticos e OTC."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from src.indicator.service import run_indicator
from src.shared.paths import REPO_ROOT


def main() -> None:
    """Carrega configuracao do indicador e inicia observacao publica."""
    cfg_path = REPO_ROOT / "config" / "settings.json"
    full_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    config = full_cfg.get("indicator", full_cfg)

    catalog_scope = config.get("catalog_scope", "stock_indices")
    if catalog_scope not in ("volatility_indices", "stock_indices"):
        raise ValueError("catalog_scope deve ser volatility_indices ou stock_indices")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    dsn = os.environ.get(
        "AETHER_TIMESCALE_DSN", config.get("timescale_dsn", "postgresql://aether:aether@127.0.0.1:5432/aether")
    )
    variant = os.environ.get("PULLBACK_VARIANT", config.get("pullback_variant", "dual_sniper"))
    models_dir = REPO_ROOT / config.get("models_dir", "data/indicator-models")
    public_ws_url = config.get("public_ws_url", "wss://api.derivws.com/trading/v1/options/ws/public")
    metrics_port = int(config.get("metrics_port", 9101))

    asyncio.run(
        run_indicator(
            dsn,
            models_dir,
            public_ws_url,
            metrics_port,
            variant=variant,
            scope=catalog_scope,
        )
    )


if __name__ == "__main__":
    main()
