"""Treino offline do indicador para o catalogo sintetico atual."""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from indicator.deriv import PublicDeriv
from indicator.pullback_backtest import format_daily_backtest
from indicator.pullback_promotion import format_ranking_table, rank_m5_pairs
from indicator.pullback_store import PullbackStore
from indicator.pullback_training import diagnose_all


async def train() -> None:
    """Prepara dados e diagnosticos do setup fixo por ativo e periodo."""
    root = Path(__file__).resolve().parent.parent
    repo_root = root.parent if root.name == "indicator" else root
    repo_cfg_path = repo_root / "config" / "settings.json"
    repo_cfg = json.loads(repo_cfg_path.read_text(encoding="utf-8")) if repo_cfg_path.is_file() else {}
    config = json.loads((root / "config" / "settings.json").read_text(encoding="utf-8"))
    if config.get("catalog_scope") not in ("volatility_indices", "stock_indices"):
        raise ValueError("catalog_scope deve ser volatility_indices ou stock_indices")
    source = PublicDeriv(config["public_ws_url"])
    store = PullbackStore(os.environ.get("AETHER_TIMESCALE_DSN", config["timescale_dsn"]))
    variant = os.environ.get("PULLBACK_VARIANT", config.get("pullback_variant", "base"))
    try:
        await store.start()
        scope = config.get("catalog_scope", "stock_indices")
        timestamp = int(time.time())
        result = await diagnose_all(source, store, now=timestamp, variant=variant, scope=scope)
        excluded = set(
            repo_cfg.get("pullback_strategy", {}).get("excluded_symbols")
            or repo_cfg.get("strategy", {}).get("excluded_symbols")
            or config.get("pullback_strategy", {}).get("excluded_symbols")
            or config.get("strategy", {}).get("excluded_symbols")
            or []
        )
        target_period = int(repo_cfg.get("pullback_strategy", {}).get("period_seconds", 900))
        payout_rate = float(
            repo_cfg.get("pullback_strategy", {}).get(
                "min_payout_rate", 0.7245 if config.get("catalog_scope") == "stock_indices" else 0.85
            )
        )
        ranked = rank_m5_pairs(result, excluded_symbols=excluded, payout=payout_rate, target_period=target_period)
        if not any(r["setups"] > 0 for r in ranked) and target_period != 900:
            ranked = rank_m5_pairs(result, excluded_symbols=excluded, payout=payout_rate, target_period=900)
        best = ranked[0] if ranked else None
        if best:
            print("\n" + format_ranking_table(ranked) + "\n")
            print("Diagnostico historico apenas: nenhuma versao foi promovida para REAL pelo treino.\n")
        print("\n" + format_daily_backtest(result, timestamp, payout_rate) + "\n")
        max_losses_global = max((int(row.get("max_losses_consecutivos", 0)) for row in result.values()), default=0)
        summary = {
            "m15_acertos_proxy": sum(row.get("acertos", 0) for key, row in result.items() if key.endswith(":900")),
            "m15_erros_proxy": sum(row.get("erros", 0) for key, row in result.items() if key.endswith(":900")),
            "m15_setups": sum(row.get("setups", 0) for key, row in result.items() if key.endswith(":900")),
            "acertos_proxy": sum(row.get("acertos", 0) for row in result.values()),
            "empates_proxy": sum(row.get("empates", 0) for row in result.values()),
            "erros_proxy": sum(row.get("erros", 0) for row in result.values()),
            "max_losses_consecutivos": 0,
            "max_losses_consecutivos_global": max_losses_global,
            "max_losses_consecutivos_promovido": 0,
            "pares": len(result),
            "pares_com_proxy": sum(row["estado"] == "PROXY DE VELAS" for row in result.values()),
            "pares_sem_sinal": sum(row["estado"] == "SEM SINAL" for row in result.values()),
            "pares_coleta_indisponivel": sum(row["estado"] == "COLETA INDISPONIVEL" for row in result.values()),
            "promovido": None,
            "qualificacao_broker": "nao realizada",
            "setups": sum(row.get("setups", 0) for row in result.values()),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        await store.close()
        await source.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    asyncio.run(train())
