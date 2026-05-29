"""CLI de backtest walk-forward Medallion M15."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from aether_paths import repo_path
from scripts.backtest.data_loader import backtest_symbols, fetch_market_for_backtest, load_settings
from scripts.backtest.gemini_collect import collect_hft_orders_gemini
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


async def run_backtest(
    config: dict[str, Any],
    market: dict,
    *,
    stake: float | None,
    bankroll: float,
    mode: str = "quant",
    gemini_cache: Path | None = None,
    max_llm_bars: int | None = None,
    llm_bar_step: int = 5,
    gemini_schedule: str = "tag_change",
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

    if mode == "gemini":
        all_orders, hft_stats = await collect_hft_orders_gemini(
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
            cache_path=str(gemini_cache) if gemini_cache else None,
            max_llm_bars=max_llm_bars,
            llm_bar_step=llm_bar_step,
            gemini_schedule=gemini_schedule,
        )
        run_mode = "gemini"
    else:
        all_orders, hft_stats = await collect_hft_orders(
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
        run_mode = "quant_surrogate"

    risk_cfg = config.get("risk_management", {})
    if stake is not None:
        sim = settle_orders(
            all_orders,
            m15,
            stake=stake,
            payout=payout,
            bankroll_start=bankroll,
            risk_config=risk_cfg if isinstance(risk_cfg, dict) else None,
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
        "mode": run_mode,
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
    parser = argparse.ArgumentParser(description="Backtest Medallion M15 (quant ou Gemini)")
    parser.add_argument("--config", type=Path, default=repo_path("config", "settings.json"))
    parser.add_argument("--output", type=Path, default=Path("data/backtest/report.json"))
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("quant", "gemini"),
        default="quant",
        help="quant=surrogate sem API; gemini=mesmo prompt/decisao do live (requer GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--gemini-cache",
        type=Path,
        default=Path("data/backtest/gemini_cache.jsonl"),
        help="Cache JSONL de respostas por bar_index (modo gemini).",
    )
    parser.add_argument(
        "--max-llm-bars",
        type=int,
        default=None,
        help="Limita chamadas Gemini (util para teste de custo).",
    )
    parser.add_argument(
        "--gemini-schedule",
        choices=("daily", "tag_change", "bar"),
        default="tag_change",
        help="tag_change=quando macro muda (padrao); daily=1 API/dia sessao; bar=cada N velas.",
    )
    parser.add_argument(
        "--llm-bar-step",
        type=int,
        default=5,
        help="So com --gemini-schedule bar: consulta a cada N velas M15.",
    )
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
    cache_path = args.gemini_cache if args.mode == "gemini" else None
    _, report = await run_backtest(
        config,
        market,
        stake=args.stake,
        bankroll=args.bankroll,
        mode=args.mode,
        gemini_cache=cache_path,
        max_llm_bars=args.max_llm_bars,
        llm_bar_step=args.llm_bar_step,
        gemini_schedule=args.gemini_schedule,
    )
    save_report(args.output, report)
    print_summary(report)
    print(f"Relatorio salvo em {args.output}")
    return 0


def main() -> None:
    """Executa CLI."""
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
