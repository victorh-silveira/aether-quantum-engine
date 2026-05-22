"""CLI de backtest walk-forward Medallion M15."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from scripts.backtest.data_loader import backtest_symbols, fetch_market_for_backtest, load_settings
from scripts.backtest.hft_cycle import collect_hft_orders
from scripts.backtest.report import build_report, print_summary, save_report
from scripts.backtest.simulator import settle_orders, settle_orders_kelly
from src.application.services.llm.macro_config import resolve_macro_config


def _min_start_bar(config: dict[str, Any]) -> int:
    """Primeiro indice M15 com historico suficiente."""
    macro = config.get("strategy", {}).get("macro", {})
    cfg = resolve_macro_config(macro if isinstance(macro, dict) else None)
    need = max(
        int(cfg.get("statarb_lookback", 30)),
        int(cfg.get("cluster_bars", 8)),
        3,
    )
    return max(need - 1, 0)


def run_backtest(
    config: dict[str, Any],
    market: dict,
    *,
    stake: float | None,
    bankroll: float,
) -> tuple[list, dict]:
    """Executa loop barra a barra e retorna trades liquidados."""
    m15 = market.get("m15", {})
    m5 = market.get("m5", {})
    us_syms, eu_syms, all_syms, anchor = backtest_symbols(config)
    macro_cfg = config.get("strategy", {}).get("macro")

    anchor_series = m15.get(anchor, [])
    if len(anchor_series) < 4:
        raise RuntimeError(f"Serie M15 insuficiente para {anchor}")

    start = _min_start_bar(config)
    end = len(anchor_series) - 2

    risk = config.get("risk_management", {})
    params = risk.get("params", {}) if isinstance(risk.get("params"), dict) else {}
    payout = float(params.get("payout_estimate", 0.95))

    all_orders, hft_stats = collect_hft_orders(
        config=config,
        m15=m15,
        m5=m5,
        us_syms=us_syms,
        eu_syms=eu_syms,
        all_syms=all_syms,
        anchor=anchor,
        macro_cfg=macro_cfg if isinstance(macro_cfg, dict) else None,
        start=start,
        end=end,
    )

    risk_cfg = config.get("risk_management", {})
    if stake is not None:
        sim = settle_orders(
            all_orders,
            m15,
            stake=stake,
            payout=payout,
            bankroll_start=bankroll,
        )
        sizing_mode = "fixed"
    else:
        sim = settle_orders_kelly(
            all_orders,
            m15,
            risk_cfg if isinstance(risk_cfg, dict) else {},
            bankroll_start=bankroll,
            payout=payout,
        )
        sizing_mode = "kelly_recovery"
    meta = {
        "mode": "quant_surrogate",
        "sizing_mode": sizing_mode,
        "stop_win_backtest": "daily_reset",
        "bars_evaluated": end - start + 1,
        "anchor": anchor,
        "symbols": all_syms,
        **hft_stats,
        **(market.get("meta") or {}),
    }
    report = build_report(sim, meta=meta, bankroll_start=bankroll, sizing_mode=sizing_mode)
    return sim.trades, report


def parse_args() -> argparse.Namespace:
    """Argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Backtest Medallion M15 (quant surrogate)")
    parser.add_argument("--config", type=Path, default=Path("config/settings.json"))
    parser.add_argument("--output", type=Path, default=Path("data/backtest/report.json"))
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument(
        "--stake",
        type=float,
        default=None,
        help="Stake fixa (opcional). Omitir para Kelly + recuperacao de risk_management.",
    )
    parser.add_argument("--bankroll", type=float, default=100.0)
    return parser.parse_args()


async def async_main() -> int:
    """Ponto de entrada assincrono."""
    args = parse_args()
    config = load_settings(args.config)
    market = await fetch_market_for_backtest(config, days=args.days, bars=args.bars)
    _, report = run_backtest(config, market, stake=args.stake, bankroll=args.bankroll)
    save_report(args.output, report)
    print_summary(report)
    print(f"Relatorio salvo em {args.output}")
    return 0


def main() -> None:
    """Executa CLI."""
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
