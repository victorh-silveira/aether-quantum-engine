"""Treino offline do indicador para o catalogo sintetico atual."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from indicator.deriv import PublicDeriv
from indicator.pullback import version_for_variant
from indicator.pullback_promotion import format_ranking_table, promote_best_m5_pair, rank_m5_pairs
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
        result = await diagnose_all(source, store, variant=variant, scope=scope)
        variant_version = version_for_variant(variant)
        excluded = set(
            repo_cfg.get("pullback_strategy", {}).get("excluded_symbols")
            or repo_cfg.get("strategy", {}).get("excluded_symbols")
            or config.get("pullback_strategy", {}).get("excluded_symbols")
            or config.get("strategy", {}).get("excluded_symbols")
            or []
        )
        target_period = int(
            repo_cfg.get("pullback_strategy", {}).get(
                "period_seconds", 900 if config.get("catalog_scope") == "stock_indices" else 300
            )
        )
        payout_rate = float(
            repo_cfg.get("pullback_strategy", {}).get(
                "min_payout_rate", 0.7245 if config.get("catalog_scope") == "stock_indices" else 0.85
            )
        )
        ranked = rank_m5_pairs(result, excluded_symbols=excluded, payout=payout_rate, target_period=target_period)
        if not any(r["setups"] > 0 for r in ranked) and target_period != 300:
            ranked = rank_m5_pairs(result, excluded_symbols=excluded, payout=payout_rate, target_period=300)
        promoted_symbol = None
        best = ranked[0] if ranked else None
        if best:
            print("\n" + format_ranking_table(ranked) + "\n")
            if best["eligible"]:
                promote_best_m5_pair(best, root, variant=variant, target_period=target_period)
                promoted_symbol = best["symbol"]
                if getattr(store, "pool", None) is not None:
                    try:
                        await store.promote_symbol_setup(best["symbol"], target_period, variant_version)
                    except Exception as exc:
                        logging.warning("Nao foi possivel atualizar pullback_registry no banco: %s", exc)
                print("=" * 88)
                print(f"[PROMOVIDO] ATIVO PROMOVIDO PARA EXECUCAO (make app-run): {best['symbol']}")
                print(
                    f"   - Taxa de Acerto: {best['win_rate'] * 100:.2f}% (Wins: {best['wins']} | Losses: {best['losses']})"
                )
                print(f"   - EV Estimado: {best['ev']:+.4f} (base payout 85%)")
                print(f"   - Maximo de Losses Consecutivos: {best.get('max_consecutive_losses', 0)}")
                print(f"   - Timeframe: M{target_period // 60} ({target_period}s / {target_period // 60}m)")
                print(f"   - Total de Setups M{target_period // 60}: {best['setups']}")
                opt_hours = best.get("optimal_hours_brt") or []
                hours_str = ", ".join(f"{h:02d}:00" for h in opt_hours) if opt_hours else "05:00 as 13:00"
                print(f"   - Melhores Horarios (Brasilia / BRT): {hours_str}")
                print(f"   - Janela Ideal de Trading: {best.get('best_window_brt', '05:00 as 13:00 (BRT)')}")
                print(f"   - Configuracao atualizada: config/settings.json (anchor: {best['symbol']})")
                print("   - Execute 'make app-run' para iniciar a operacao no ativo promovido.")
                print("=" * 88 + "\n")
        max_losses_promovido = int(best.get("max_consecutive_losses", 0)) if best and best.get("eligible") else 0
        max_losses_global = max((int(row.get("max_losses_consecutivos", 0)) for row in result.values()), default=0)
        summary = {
            "acertos_proxy": sum(row.get("acertos", 0) for row in result.values()),
            "empates_proxy": sum(row.get("empates", 0) for row in result.values()),
            "erros_proxy": sum(row.get("erros", 0) for row in result.values()),
            "max_losses_consecutivos": max_losses_promovido,
            "max_losses_consecutivos_global": max_losses_global,
            "max_losses_consecutivos_promovido": max_losses_promovido,
            "pares": len(result),
            "pares_com_proxy": sum(row["estado"] == "PROXY DE VELAS" for row in result.values()),
            "pares_sem_sinal": sum(row["estado"] == "SEM SINAL" for row in result.values()),
            "promovido": promoted_symbol,
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
