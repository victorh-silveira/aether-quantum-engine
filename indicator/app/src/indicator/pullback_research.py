"""Comparacao cronologica de variantes fixas sem promocao automatica."""

from __future__ import annotations

from indicator.domain import Candle
from indicator.pullback import MIN_BARS, evaluate

VARIANTS = (
    "base",
    "ema50_slope",
    "countertrend_retrace",
    "entry_distance",
    "improved",
    "momentum_burst",
    "senior_institutional",
)


def compare_variants(candles: list[Candle], period: int) -> dict[str, dict]:
    """Conta setups e proxies em janelas anteriores e teste final intocado."""
    if len(candles) < 5 * MIN_BARS:
        return {}
    n = len(candles)
    windows = ((0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))
    report: dict[str, dict] = {}
    for variant in VARIANTS:
        results = []
        for start_fraction, end_fraction in windows:
            end = min(n - 1, int(n * end_fraction))
            start = max(MIN_BARS - 1, int(n * start_fraction), end - 250)
            wins = losses = ties = 0
            for index in range(start, end):
                bars = candles[index - MIN_BARS + 1 : index + 1]
                setup = evaluate(bars, period, candles[index].epoch + period, variant)
                if setup.side not in ("CALL", "PUT"):
                    continue
                next_bar = candles[index + 1]
                delta = next_bar.close - next_bar.open
                if delta == 0:
                    ties += 1
                elif (delta > 0) == (setup.side == "CALL"):
                    wins += 1
                else:
                    losses += 1
            results.append(
                {"setups": wins + losses + ties, "acertos_proxy": wins, "erros_proxy": losses, "empates": ties}
            )
        report[variant] = {"janelas": results, "teste_final": results[-1], "qualifica_contrato": False}
    return report
