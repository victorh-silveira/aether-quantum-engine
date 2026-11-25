"""Gera sinais causais comparaveis e audita equivalencia com avaliadores escalares."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from scripts.analysis.frequency_features import batch_features
from scripts.analysis.frequency_replay import Signal
from src.indicator.domain import Candle
from src.indicator.frequency_candidate import Candidate, evaluate_candidate, generate_candidates
from src.indicator.pullback import evaluate


def read_dataset(path: Path) -> dict[tuple[str, int], list[Candle]]:
    """Le apenas as colunas OHLC canonicas em ordem temporal."""
    frame = pl.read_parquet(path).sort(["symbol", "period", "epoch"])
    return {
        (symbol, period): [Candle(*row) for row in part.select("epoch", "open", "high", "low", "close").iter_rows()]
        for (symbol, period), part in frame.partition_by(["symbol", "period"], as_dict=True).items()
    }


def candidate_sides(features: dict[str, np.ndarray], candidate: Candidate) -> np.ndarray:
    """Traduz exatamente os cinco ramos para vetores, mantendo veto de conflito."""
    scale = candidate.di_scale
    h1 = features["h1_trend"]
    direction = features["direction"]
    signals = []
    for branch in candidate.branches:
        if branch == 1:
            di = features["di24"]
            mask = (abs(di) > 40 * scale) & (np.sign(di) == h1)
        elif branch == 2:
            di = features["di14"]
            mask = (abs(di) > 45 * scale) & (np.sign(di) == h1)
        elif branch == 3:
            di = features["di14"]
            mask = (abs(di) > 45 * scale) & (np.sign(di) == -direction)
        elif branch == 4:
            di = features["di28"]
            mask = (abs(di) > 38 * scale) & (np.sign(di) == direction)
        else:
            di = features["di20"]
            mask = (abs(di) > 40 * scale) & (features["adx20"] >= candidate.adx_min)
        signals.append(np.where(mask, np.sign(di), 0).astype(np.int8))
    stack = np.array(signals)
    positive = np.any(stack == 1, axis=0)
    negative = np.any(stack == -1, axis=0)
    return np.where(positive & ~negative, 1, np.where(negative & ~positive, -1, 0)).astype(np.int8)


def _usable(bars: list[Candle], cutoff: int, start: int) -> tuple[np.ndarray, dict]:
    """Exige entrada e resultado contiguos sem imputar fechamento de sessoes."""
    epoch = np.array([bar.epoch for bar in bars])
    next_epoch = np.roll(epoch, -1)
    prev_epoch = np.roll(epoch, 1)
    eligible = (np.arange(len(bars)) >= 50) & (epoch + 900 >= start) & (epoch + 1800 <= cutoff)
    gaps = (epoch - prev_epoch != 900) | (next_epoch - epoch != 900)
    eligible &= ~gaps
    eligible[-1:] = False
    return eligible, {
        "candles": len(bars),
        "contiguous_evaluations": int(eligible.sum()),
        "gap_or_missing_result": int(gaps.sum()),
    }


def _signals(symbol: str, bars: list[Candle], sides: np.ndarray) -> list[Signal]:
    """Usa resultado apenas para rotular sinais ja definidos causalmente."""
    result = []
    for index in np.flatnonzero(sides):
        nxt = bars[index + 1]
        outcome = int(np.sign(nxt.close - nxt.open)) * int(sides[index])
        result.append(Signal(symbol, bars[index].epoch + 900, "CALL" if sides[index] == 1 else "PUT", outcome))
    return result


def persistent_signals(dataset: dict, start: int, cutoff: int) -> tuple[dict[str, list[Signal]], dict]:
    """Calcula 145 hipoteses fixas e confere paridade em amostra deterministica."""
    candidates = generate_candidates()
    output = {candidate.label: [] for candidate in candidates}
    audit = {}
    for (symbol, period), bars in dataset.items():
        if period != 900 or not bars:
            continue
        h1 = dataset.get((symbol, 3600), [])
        h1_epochs = [b.epoch for b in h1]
        features = batch_features(bars, h1)
        usable, inventory = _usable(bars, cutoff, start)
        mask = usable & ~features["flat"] & ~features["toxic"]
        reasons = Counter(
            {
                "hora_toxica_utc": int((usable & features["toxic"] & ~features["flat"]).sum()),
                "mercado_sem_oscilacao": int((usable & features["flat"]).sum()),
            }
        )
        checks_per_candidate = 0
        for candidate in candidates:
            sides = np.where(mask, candidate_sides(features, candidate), 0)
            output[candidate.label].extend(_signals(symbol, bars, sides))
            indices = np.flatnonzero(usable)
            check = indices[np.linspace(0, len(indices) - 1, min(8, len(indices)), dtype=int)] if len(indices) else []
            checks_per_candidate = len(check)
            for i in check:
                end = bisect_right(h1_epochs, bars[i].epoch + 900 - 3600)
                actual = evaluate_candidate(bars[max(0, i - 499) : i + 1], h1[max(0, end - 500) : end], candidate)[0]
                expected = "CALL" if sides[i] == 1 else "PUT" if sides[i] == -1 else "SEM SINAL"
                if actual != expected:
                    raise ValueError(f"Paridade escalar falhou: {symbol} {candidate.label} {bars[i].epoch}")
            if candidate == Candidate():
                reasons["setup_confirmado"] = int(np.count_nonzero(sides))
                reasons["sem_setup_persistent_trend"] = int(mask.sum() - np.count_nonzero(sides))
        audit[symbol] = {
            **inventory,
            "baseline_reasons": dict(reasons),
            "scalar_parity_checks": checks_per_candidate * len(candidates),
        }
        print(f"FEATURES {symbol}: {len(bars)} velas; paridade escalar verificada", flush=True)
    return output, audit


def comparator_signals(dataset: dict, start: int, cutoff: int) -> dict[str, list[Signal]]:
    """Reexecuta variantes existentes com pares e H1 estritamente causais."""
    variants = ("dual_sniper", "multi_regime_hf", "senior_institutional")
    output = {name: [] for name in variants}
    peers = {symbol: bars for (symbol, period), bars in dataset.items() if period == 900}
    peer_epochs = {symbol: [b.epoch for b in bars] for symbol, bars in peers.items()}
    for symbol, bars in peers.items():
        h1 = dataset.get((symbol, 3600), [])
        h1_epochs = [b.epoch for b in h1]
        usable, _ = _usable(bars, cutoff, start)
        for i in np.flatnonzero(usable):
            epoch = bars[i].epoch
            peer_slice = {}
            for peer, series in peers.items():
                end = bisect_right(peer_epochs[peer], epoch)
                if end and series[end - 1].epoch == epoch:
                    peer_slice[peer] = series[max(0, end - 500) : end]
            h_end = bisect_right(h1_epochs, epoch + 900 - 3600)
            for variant in variants:
                result = evaluate(
                    bars[max(0, i - 499) : i + 1],
                    900,
                    epoch + 900,
                    variant,
                    symbol=symbol,
                    peer_bars=peer_slice,
                    h1_bars=h1[max(0, h_end - 500) : h_end],
                )
                if result.side in ("CALL", "PUT"):
                    outcome = int(np.sign(bars[i + 1].close - bars[i + 1].open)) * (1 if result.side == "CALL" else -1)
                    output[variant].append(Signal(symbol, epoch + 900, result.side, outcome))
        print(f"COMPARADORES {symbol} concluidos", flush=True)
    return output


def session_gap_signals(dataset: dict, start: int, cutoff: int) -> list[dict]:
    """Identifica sinais que a tolerancia legada a gaps aceita e a pesquisa exclui."""
    output = []
    for (symbol, period), bars in dataset.items():
        if period != 900:
            continue
        h1 = dataset.get((symbol, 3600), [])
        h1_epochs = [b.epoch for b in h1]
        for i in range(50, len(bars) - 1):
            epoch = bars[i].epoch
            previous_gap = epoch - bars[i - 1].epoch
            if not start <= epoch + 900 < cutoff or previous_gap == 900 or bars[i + 1].epoch != epoch + 900:
                continue
            h_end = bisect_right(h1_epochs, epoch - 2700)
            result = evaluate(
                bars[max(0, i - 499) : i + 1],
                900,
                epoch + 900,
                "persistent_trend_v3",
                allow_session_gaps=True,
                symbol=symbol,
                h1_bars=h1[max(0, h_end - 500) : h_end],
            )
            if result.side in ("CALL", "PUT"):
                outcome = int(np.sign(bars[i + 1].close - bars[i + 1].open)) * (1 if result.side == "CALL" else -1)
                output.append(
                    {
                        "symbol": symbol,
                        "entry_epoch": epoch + 900,
                        "side": result.side,
                        "outcome": outcome,
                        "previous_gap_seconds": previous_gap,
                    }
                )
    return output
