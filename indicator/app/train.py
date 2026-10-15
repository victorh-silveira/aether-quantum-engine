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
    config = json.loads((root / "config" / "settings.json").read_text(encoding="utf-8"))
    if config.get("catalog_scope") != "volatility_indices":
        raise ValueError("catalog_scope deve ser volatility_indices")
    source = PublicDeriv(config["public_ws_url"])
    store = PullbackStore(os.environ.get("AETHER_TIMESCALE_DSN", config["timescale_dsn"]))
    variant = os.environ.get("PULLBACK_VARIANT", config.get("pullback_variant", "base"))
    try:
        await store.start()
        result = await diagnose_all(source, store, variant=variant)
        variant_version = version_for_variant(variant)
        excluded = set(
            config.get("pullback_strategy", {}).get("excluded_symbols")
            or config.get("strategy", {}).get("excluded_symbols")
            or []
        )
        ranked = rank_m5_pairs(result, excluded_symbols=excluded)
        promoted_symbol = None
        if ranked:
            print("\n" + format_ranking_table(ranked) + "\n")
            best = ranked[0]
            if best["eligible"]:
                promote_best_m5_pair(best, root, variant=variant)
                promoted_symbol = best["symbol"]
                if getattr(store, "pool", None) is not None:
                    try:
                        await store.promote_symbol_setup(best["symbol"], 300, variant_version)
                    except Exception as exc:
                        logging.warning("Nao foi possivel atualizar pullback_registry no banco: %s", exc)
                print("=" * 88)
                print(f"[PROMOVIDO] ATIVO PROMOVIDO PARA EXECUCAO (make app-run): {best['symbol']}")
                print(
                    f"   - Taxa de Acerto: {best['win_rate'] * 100:.2f}% (Wins: {best['wins']} | Losses: {best['losses']})"
                )
                print(f"   - EV Estimado: {best['ev']:+.4f} (base payout 85%)")
                print(f"   - Total de Setups M5: {best['setups']}")
                print(f"   - Configuracao atualizada: config/settings.json (anchor: {best['symbol']})")
                print("   - Execute 'make app-run' para iniciar a operacao no ativo promovido.")
                print("=" * 88 + "\n")
        summary = {
            "pares": len(result),
            "pares_com_proxy": sum(row["estado"] == "PROXY DE VELAS" for row in result.values()),
            "pares_sem_sinal": sum(row["estado"] == "SEM SINAL" for row in result.values()),
            "setups": sum(row.get("setups", 0) for row in result.values()),
            "acertos_proxy": sum(row.get("acertos", 0) for row in result.values()),
            "erros_proxy": sum(row.get("erros", 0) for row in result.values()),
            "empates_proxy": sum(row.get("empates", 0) for row in result.values()),
            "promovido": promoted_symbol,
            "qualificacao_broker": "nao realizada",
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        await store.close()
        await source.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    asyncio.run(train())
