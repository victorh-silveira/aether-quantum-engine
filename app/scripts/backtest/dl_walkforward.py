"""CLI de backtest walk-forward Deep Learning TCN."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from aether_paths import repo_path
from scripts.backtest.dl_walkforward_fetch import fetch_m1_closes
from src.application.services.deep_learning.dl_bridge_helpers import parse_dl_params
from src.application.services.deep_learning.dl_sim_backtest import DlBacktestResult, run_dl_walkforward


def load_settings(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def print_dl_summary(result: DlBacktestResult, *, symbol: str) -> None:
    print(f"DL walk-forward | symbol={symbol}")
    print(f"  trades={len(result.trades)} win_rate={result.win_rate:.3f} PF={result.profit_factor:.2f}")
    print(f"  max_dd={result.max_drawdown:.2f} trades/dia={result.trades_per_day:.1f}")
    print(f"  val_brier={result.val_brier:.3f} deploy_ok={result.deploy_ok}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest walk-forward Deep Learning TCN")
    parser.add_argument("--config", type=Path, default=repo_path("config", "settings.json"))
    parser.add_argument("--symbol", type=str, default="1HZ100V")
    parser.add_argument("--bars", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("data/backtest/dl_report.json"))
    parser.add_argument("--retrain-every", type=int, default=120)
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    config = load_settings(args.config)
    params = parse_dl_params(config.get("deep_learning", {}))
    prices_list = await fetch_m1_closes(args.symbol, args.bars, config)
    if len(prices_list) < 100:
        raise RuntimeError(f"Serie M1 insuficiente para {args.symbol}: {len(prices_list)} velas")
    prices = np.asarray(prices_list, dtype=np.float64)
    result = run_dl_walkforward(prices, params, retrain_every=args.retrain_every)
    print_dl_summary(result, symbol=args.symbol)
    payload = {
        "symbol": args.symbol,
        "bars": len(prices_list),
        "trades": len(result.trades),
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "max_drawdown": result.max_drawdown,
        "trades_per_day": result.trades_per_day,
        "val_brier": result.val_brier,
        "deploy_ok": result.deploy_ok,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Relatorio salvo em {args.output}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
