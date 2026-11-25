"""Inferencia temporal do proxy OHLC sem promocao automatica de contratos."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np


BRT = timezone(timedelta(hours=-3))
PAYOUTS = (0.70, 0.7235, 0.80)
SEED = 42


def daily_counts(trades: list[dict], start_day: str, end_day: str) -> tuple[list[str], np.ndarray]:
    """Mantem dias zerados e dependencia entre todos os ativos de cada dia."""
    first = datetime.fromisoformat(start_day).date()
    last = datetime.fromisoformat(end_day).date()
    days = [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]
    index = {day: i for i, day in enumerate(days)}
    counts = np.zeros((len(days), 3), dtype=np.int64)
    for trade in trades:
        day = datetime.fromtimestamp(trade["entry_epoch"], tz=BRT).date().isoformat()
        if day in index:
            outcome = int(trade["outcome"])
            counts[index[day], {1: 0, -1: 1, 0: 2}[outcome]] += 1
    return days, counts


def infer_ev(counts: np.ndarray, *, payout: float = 0.70, samples: int = 10000, seed: int = SEED) -> dict:
    """Bootstrap circular centrado de blocos de cinco dias para EV por ordem."""
    n_days = len(counts)
    totals = counts.sum(axis=0) if n_days else np.zeros(3)
    n = int(totals.sum())
    active = int(np.count_nonzero(counts.sum(axis=1))) if n_days else 0
    if n == 0 or n_days == 0:
        return {"n": n, "active_days": active, "ev": None, "lower95": None, "p_value": 1.0}
    returns = counts[:, 0] * payout - counts[:, 1]
    orders = counts.sum(axis=1)
    mean = float(returns.sum() / n)
    centered = returns - mean * orders
    rng = np.random.default_rng(seed)
    block_count = (n_days + 4) // 5
    starts = rng.integers(0, n_days, size=(samples, block_count))
    indices = ((starts[:, :, None] + np.arange(5)) % n_days).reshape(samples, -1)[:, :n_days]
    denominator = orders[indices].sum(axis=1)
    valid = denominator > 0
    deviations = centered[indices[valid]].sum(axis=1) / denominator[valid]
    lower = mean - float(np.quantile(deviations, 0.95)) if len(deviations) else mean
    p_value = (1 + int(np.count_nonzero(deviations >= mean))) / (1 + len(deviations))
    return {"n": n, "active_days": active, "ev": mean, "lower95": lower, "p_value": p_value}


def holm_adjust(results: list[dict]) -> list[dict]:
    """Ajusta a familia completa de tentativas sem excluir candidatos ruins."""
    order = sorted(range(len(results)), key=lambda i: results[i]["p_value"])
    previous = 0.0
    for rank, index in enumerate(order):
        row = results[index]
        adjusted = max(previous, min(1.0, row["p_value"] * (len(order) - rank)))
        previous = adjusted
        row["holm_p"] = adjusted
        row["statistically_eligible"] = bool(
            row["n"] >= 200
            and row["active_days"] >= 30
            and row["lower95"] is not None
            and row["lower95"] > 0
            and adjusted <= 0.05
        )
    return results


def select_candidate(results: list[dict], *, data_qualified: bool) -> dict | None:
    """Seleciona frequencia apenas apos gates estatisticos e integridade de dados."""
    eligible = [row for row in results if row.get("statistically_eligible") and data_qualified]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["trades_per_day"], row["lower95"], -row["max_drawdown"]))
