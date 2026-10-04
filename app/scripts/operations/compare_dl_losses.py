"""Compara perdas da TCN em OHLC da granularidade ativa sem exportar checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path

import asyncpg
import numpy as np
import torch


_APP_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _APP_ROOT.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_calibration import apply_calibrator_stable
from src.application.services.deep_learning.dl_model_factory import create_direction_model
from src.application.services.deep_learning.dl_sequence_extract import extract_sequences_and_deltas
from src.application.services.deep_learning.dl_splits import purged_temporal_splits
from src.application.services.deep_learning.dl_training import train_model_walkforward
from src.application.services.deep_learning.model import _model_raw_prob, normalize_sequences


async def load_ohlc(
    dsn: str, symbol: str, bars: int, granularity: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Le uma janela OHLC da granularidade ativa contigua do Timescale sem alterar o banco."""
    conn = await asyncpg.connect(dsn, timeout=10.0)
    try:
        rows = await conn.fetch(
            "SELECT epoch, open, high, low, close FROM ohlc_bars "
            "WHERE symbol=$1 AND granularity=$3 AND epoch % $3=0 "
            "ORDER BY epoch DESC LIMIT $2",
            symbol,
            bars,
            granularity,
        )
    finally:
        await conn.close()
    rows = list(reversed(rows))
    if len(rows) < 1000:
        raise ValueError(f"OHLC da granularidade ativa insuficiente: {len(rows)} barras")
    epochs = np.asarray([row["epoch"] for row in rows], dtype=np.int64)
    if np.any(np.diff(epochs) != granularity):
        raise ValueError("OHLC da granularidade ativa possui lacunas; comparacao temporal interrompida")
    arrays = tuple(np.asarray([row[key] for row in rows], dtype=np.float64) for key in ("close", "open", "high", "low"))
    if not all(np.isfinite(arr).all() for arr in arrays):
        raise ValueError("OHLC da granularidade ativa contem valores nao finitos")
    return arrays


def compare_losses(
    settings: dict,
    ohlc: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    seed: int,
    epochs: int,
) -> list[dict[str, float | int | str]]:
    """Treina duas TCNs com mesma inicializacao e splits, sem persistencia."""
    dl = settings["deep_learning"]
    prices, open_, high, low = ohlc
    granularity = int(settings["data_handler"]["micro_granularity"])
    cfg = {**dl, "training_device": "cpu", "inference_device": "cpu"}
    tcn = dl["tcn"]
    x_all, y_all, mask_all, _ = extract_sequences_and_deltas(
        prices,
        int(dl["lookback"]),
        granularity=granularity,
        label_horizon_bars=int(dl["label_horizon_bars"]),
        label_smooth_bars=int(dl["label_smooth_bars"]),
        label_mode=str(dl["label_mode"]),
        label_ma_window=int(dl["label_ma_window"]),
        implied_vol_bars=int(dl["implied_vol_bars"]),
        symbol=str(settings["anchor"]),
        open_=open_,
        high=high,
        low=low,
    )
    active_idx = np.flatnonzero(mask_all > 0.5)
    validation_bars = int(len(prices) * float(dl["validation_ratio"]))
    splits = purged_temporal_splits(
        len(active_idx),
        min(validation_bars, max(5, int(len(active_idx) * 0.20))),
        calib_ratio=float(dl["calib_ratio"]),
        embargo=max(1, int(dl["label_horizon_bars"]) + int(dl["label_smooth_bars"]) - 1),
        stride=max(1, int(dl["label_horizon_bars"])),
    )
    if splits is None:
        raise ValueError("amostras ativas insuficientes para comparacao")
    train_sl, val_sl, _ = splits
    val_active = active_idx[val_sl]
    full_window = slice(int(val_active[0]), int(val_active[-1]) + 1)
    train_rate = float(np.mean(y_all[active_idx[train_sl]]))
    constant_call = train_rate >= 0.5
    full_y = y_all[full_window]
    results: list[dict[str, float | int | str]] = [
        {
            "objective": "constant_train_rate",
            "seed": seed,
            "bars": len(prices),
            "active_samples": len(active_idx),
            "full_window_n": len(full_y),
            "active_window_n": len(val_active),
            "epochs": 0,
            "val_accuracy": round(float(np.mean(y_all[val_active] == constant_call)), 4),
            "val_brier": round(float(np.mean((train_rate - y_all[val_active]) ** 2)), 4),
            "val_ece": round(abs(train_rate - float(np.mean(y_all[val_active]))), 4),
            "full_window_accuracy": round(float(np.mean(full_y == constant_call)), 4),
            "full_window_brier": round(float(np.mean((train_rate - full_y) ** 2)), 4),
        }
    ]
    for name, asymmetric in (("payout_weighted", True), ("bce", False)):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = create_direction_model(
            arch="tcn",
            input_dim=FEATURE_DIM,
            tcn_channels=tuple(int(v) for v in tcn["channels"]),
            tcn_dropout=float(tcn["dropout"]),
        )
        result = train_model_walkforward(
            model,
            prices,
            lookback=int(dl["lookback"]),
            epochs=epochs,
            lr=float(dl["learning_rate"]),
            validation_bars=validation_bars,
            weight_decay=float(dl["weight_decay"]),
            calib_ratio=float(dl["calib_ratio"]),
            granularity=granularity,
            label_horizon_bars=int(dl["label_horizon_bars"]),
            label_smooth_bars=int(dl["label_smooth_bars"]),
            label_mode=str(dl["label_mode"]),
            label_ma_window=int(dl["label_ma_window"]),
            implied_vol_bars=int(dl["implied_vol_bars"]),
            symbol=str(settings["anchor"]),
            open_=open_,
            high=high,
            low=low,
            batch_size=int(dl["training_batch_size"]),
            dl_config=cfg,
            asymmetric_payout_loss=asymmetric,
        )
        if result is None:
            raise ValueError(f"treino {name} sem amostras suficientes")
        full_x = normalize_sequences(x_all[full_window], result.norm_stats)
        full_prob = np.asarray(
            [apply_calibrator_stable(float(p), result.calibrator) for p in _model_raw_prob(model, full_x)],
            dtype=np.float64,
        )
        results.append(
            {
                "objective": name,
                "seed": seed,
                "bars": len(prices),
                "active_samples": len(active_idx),
                "full_window_n": len(full_y),
                "active_window_n": len(val_active),
                "epochs": result.epochs_ran,
                "val_accuracy": round(result.val_accuracy, 4),
                "val_brier": round(result.val_brier, 4),
                "val_ece": round(result.val_ece, 4),
                "full_window_accuracy": round(float(np.mean((full_prob >= 0.5) == full_y)), 4),
                "full_window_brier": round(float(np.mean((full_prob - full_y) ** 2)), 4),
            }
        )
    return results


def main() -> int:
    """Executa comparacao offline com mesmo historico para as duas perdas."""
    parser = argparse.ArgumentParser(description="Compara TCN payout weighted e BCE sem alterar checkpoint")
    parser.add_argument("--symbol", default="1HZ75V")
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    settings = json.loads((_REPO_ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
    dsn = os.environ.get("AETHER_TIMESCALE_DSN") or settings["infra"]["timescale"]["dsn"]
    granularity = int(settings["data_handler"]["micro_granularity"])
    ohlc = asyncio.run(load_ohlc(dsn, args.symbol, args.bars, granularity))
    settings["anchor"] = args.symbol
    logging.getLogger("AETH").setLevel(logging.WARNING)
    for item in compare_losses(settings, ohlc, seed=args.seed, epochs=args.epochs):
        print(json.dumps(item, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
