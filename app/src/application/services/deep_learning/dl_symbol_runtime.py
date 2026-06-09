"""Runtime de modelo, treino walk-forward e checkpoints por simbolo."""

import logging
from pathlib import Path

import numpy as np

from aether_paths import APP_ROOT
from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_deploy import apply_deploy_to_runtime
from src.application.services.deep_learning.dl_deploy_eval import evaluate_mini_deploy
from src.application.services.deep_learning.dl_features import FEATURE_DIM, extract_sequences
from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config, resolve_deploy_ok
from src.application.services.deep_learning.dl_outcomes import sample_weights_for_symbol
from src.application.services.deep_learning.dl_retrain import clear_force_retrain, reset_bars_since_train
from src.application.services.deep_learning.dl_training import train_model_walkforward
from src.application.services.deep_learning.model import (
    create_direction_model,
    fit_norm_stats,
    load_model_checkpoint,
    save_model_checkpoint,
)
from src.domain.symbols.range_symbols import hedge_peer, sym_is_low_barrier


logger = logging.getLogger("AETH")


def resolve_dl_model_path(dl_config: dict, symbol: str) -> Path:
    """Resolve caminho do checkpoint PyTorch para um simbolo."""
    template = dl_config.get("model_path_template")
    if template:
        rel = str(template).format(symbol=symbol)
        return (APP_ROOT / rel).resolve()
    legacy = dl_config.get("model_path", "data/deep_learning_model.pth")
    return (APP_ROOT / legacy).resolve()


def granularity_seconds(orch) -> int:
    """Retorna granularidade OHLC em segundos."""
    return int(orch.config.get("data_handler", {}).get("granularity", 300))


def pair_prices_for_symbol(orch, symbol: str) -> np.ndarray | None:
    """Retorna serie do simbolo par hedge R_* quando aplicavel."""
    symbols = {str(s) for s in getattr(orch, "symbols", [])}
    peer = hedge_peer(str(symbol))
    if peer is None or peer not in symbols:
        return None
    return orch.stream.get_numpy_series(peer, "close")


def get_symbol_runtime(orch, symbol: str, dl_config: dict, params: dict) -> dict:
    """Carrega ou inicializa estado de modelo e normalizacao por simbolo."""
    if not hasattr(orch, "_dl_runtime"):
        orch._dl_runtime = {}
    if symbol not in orch._dl_runtime:
        path = resolve_dl_model_path(dl_config, symbol)
        loaded = load_model_checkpoint(path)
        calibrator = CalibratorState()
        lookback = int(params["lookback"])
        deploy_ok = False
        deploy_win_rate = 0.0
        if loaded is not None:
            (
                model,
                norm_stats,
                last_epoch,
                calibrator,
                lookback,
                val_accuracy,
                val_brier,
                val_ece,
                deploy_ok,
                deploy_win_rate,
            ) = loaded
            logger.debug("DL: Checkpoint TCN carregado para %s em %s", symbol, path)
        else:
            model = create_direction_model(arch=params["arch"], input_dim=FEATURE_DIM)
            norm_stats = fit_norm_stats(np.zeros((1, lookback, FEATURE_DIM), dtype=np.float32))
            last_epoch = 0
            val_accuracy = 0.0
            val_brier = 1.0
            val_ece = 1.0
        orch._dl_runtime[symbol] = {
            "model": model,
            "norm_stats": norm_stats,
            "last_candle_epoch": last_epoch,
            "val_accuracy": val_accuracy,
            "calibrator": calibrator,
            "val_brier": val_brier,
            "val_ece": val_ece,
            "lookback": lookback,
            "deploy_ok": deploy_ok,
            "deploy_win_rate": deploy_win_rate,
        }
    return orch._dl_runtime[symbol]


def candle_epoch(orch, symbol: str) -> int:
    """Obtem epoch da ultima vela disponivel no stream."""
    getter = getattr(orch.stream, "get_last_candle_epoch", None)
    if callable(getter):
        epoch = getter(symbol)
        return int(epoch) if epoch is not None else 0
    return 0


def run_symbol_training(
    symbol: str,
    runtime: dict,
    prices: np.ndarray,
    dl_config: dict,
    params: dict,
    candle_epoch_value: int,
    orch,
    *,
    pair_prices: np.ndarray | None,
    granularity: int,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> tuple[object, float | None]:
    """Executa treino walk-forward, deploy gate e persiste checkpoint."""
    model = runtime["model"]
    norm_stats = runtime["norm_stats"]
    train_loss = None
    gate_cfg = parse_deploy_gate_config(dl_config)
    try:
        pair_label = pair_prices is not None and len(pair_prices) >= len(prices)
        peer_sym = hedge_peer(str(symbol))
        sym_is_bull = sym_is_low_barrier(str(symbol), peer_sym) if peer_sym else False
        horizon = int(params.get("label_horizon_bars", 1))
        x_preview, y_preview, _ = extract_sequences(
            prices,
            params["lookback"],
            label_min_move_pct=params["label_min_move_pct"],
            granularity=granularity,
            pair_prices=pair_prices,
            require_pair_label=pair_label,
            sym_is_bull=sym_is_bull,
            label_horizon_bars=horizon,
            open_=open_,
            high=high,
            low=low,
        )
        weights = sample_weights_for_symbol(
            orch,
            symbol,
            len(y_preview),
            targets=list(y_preview) if len(y_preview) else None,
        )
        train_result = train_model_walkforward(
            model,
            prices,
            params["lookback"],
            params["epochs"],
            params["lr"],
            params["validation_bars"],
            sample_weights=weights,
            weight_decay=params["weight_decay"],
            label_smoothing=params["label_smoothing"],
            label_min_move_pct=params["label_min_move_pct"],
            early_stopping_patience=params["early_stopping_patience"],
            focal_gamma=params["focal_gamma"],
            calib_ratio=params["calib_ratio"],
            granularity=granularity,
            pair_prices=pair_prices,
            require_pair_label=pair_label,
            sym_is_bull=sym_is_bull,
            label_horizon_bars=int(params.get("label_horizon_bars", 1)),
            open_=open_,
            high=high,
            low=low,
        )
        if train_result is not None:
            runtime["norm_stats"] = train_result.norm_stats
            norm_stats = train_result.norm_stats
            runtime["val_accuracy"] = train_result.val_accuracy
            runtime["calibrator"] = train_result.calibrator or CalibratorState()
            runtime["val_brier"] = train_result.val_brier
            runtime["val_ece"] = train_result.val_ece
            train_loss = train_result.avg_loss
            runtime["last_candle_epoch"] = candle_epoch_value
            mini_ok, deploy_wr, mini_brier = evaluate_mini_deploy(
                orch,
                symbol,
                model,
                prices,
                norm_stats,
                runtime,
                params,
                gate_cfg=gate_cfg,
            )
            deploy_ok = resolve_deploy_ok(
                mini_ok=mini_ok,
                val_accuracy=float(train_result.val_accuracy),
                val_brier=float(train_result.val_brier),
                gate_cfg=gate_cfg,
            )
            apply_deploy_to_runtime(
                runtime,
                deploy_ok=deploy_ok,
                deploy_win_rate=deploy_wr,
                val_brier=mini_brier if mini_ok else float(train_result.val_brier),
            )
            path = resolve_dl_model_path(dl_config, symbol)
            save_model_checkpoint(
                path,
                model,
                norm_stats,
                candle_epoch_value,
                lookback=params["lookback"],
                calibrator=runtime["calibrator"],
                arch=params["arch"],
                val_accuracy=runtime["val_accuracy"],
                val_brier=runtime["val_brier"],
                val_ece=runtime["val_ece"],
                deploy_ok=runtime["deploy_ok"],
                deploy_win_rate=runtime["deploy_win_rate"],
                granularity=granularity,
            )
            clear_force_retrain(orch, symbol)
            reset_bars_since_train(orch, symbol)
            logger.debug(
                "DL: Treino %s concluido | loss=%.4f val_acc=%.2f deploy=%s",
                symbol,
                float(train_loss or 0.0),
                float(runtime.get("val_accuracy", 0.0)),
                bool(runtime.get("deploy_ok", False)),
            )
        else:
            runtime["val_accuracy"] = 0.0
            runtime["val_brier"] = 1.0
            runtime["deploy_ok"] = False
            logger.debug(
                "DL: Treino indisponivel para %s (%d velas); usando checkpoint ou predicao direta.",
                symbol,
                len(prices),
            )
    except Exception as e:
        logger.error("DL: Erro no treinamento walk-forward para %s: %s", symbol, e)
        runtime["deploy_ok"] = False
    return norm_stats, train_loss
