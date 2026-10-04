"""Persistencia do checkpoint tecnico apos treino temporal."""

import logging
import time

import numpy as np

from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_horizon import contract_duration_seconds
from src.application.services.deep_learning.dl_retrain import clear_force_retrain, reset_bars_since_train
from src.application.services.deep_learning.dl_symbol_runtime import resolve_dl_model_path
from src.application.services.deep_learning.model import save_model_checkpoint
from src.application.services.live_signal_metrics import live_signal_snapshot


logger = logging.getLogger("AETH")


def _log_horizon_gap(*, level: int, symbol: str, granularity: int, params: dict, orch) -> None:
    """Registra alinhamento do horizonte do label com a duracao do contrato."""
    label_bars = max(1, int(params.get("label_horizon_bars", 1)))
    risk = getattr(orch, "config", {}).get("risk_management", {}) if orch is not None else {}
    risk_params = risk.get("params", {}) if isinstance(risk, dict) else {}
    contract_seconds = contract_duration_seconds(risk_params) if risk_params else 0
    logger.log(
        level,
        "DL TREINO | %s | horizonte label=%ds | contrato=%ds | gap=%ds",
        symbol,
        label_bars * granularity,
        contract_seconds,
        label_bars * granularity - contract_seconds,
    )


def apply_successful_symbol_train(
    symbol: str,
    runtime: dict,
    train_result,
    *,
    orch,
    model,
    prices: np.ndarray,
    norm_stats,
    params: dict,
    dl_config: dict,
    candle_epoch_value: int,
    granularity: int,
    level: int,
    started: float,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro=None,
) -> tuple[object, float]:
    """Salva modelo compativel para uso limitado sem qualificacao estatistica."""
    _ = (prices, norm_stats, open_, high, low, micro)
    runtime["norm_stats"] = train_result.norm_stats
    runtime["val_accuracy"] = train_result.val_accuracy
    runtime["calibrator"] = train_result.calibrator or CalibratorState()
    runtime["val_brier"] = train_result.val_brier
    runtime["val_ece"] = train_result.val_ece
    runtime["last_candle_epoch"] = candle_epoch_value
    runtime["deploy_ok"] = False
    runtime["deploy_provisional_ok"] = False
    runtime["deploy_win_rate"] = 0.0
    runtime["label_call_frac"] = float(getattr(train_result, "label_call_frac", 0.5))
    runtime["pred_call_frac"] = float(getattr(train_result, "pred_call_frac", 0.5))
    runtime["minority_recall"] = float(getattr(train_result, "minority_recall", 1.0))
    _log_horizon_gap(level=level, symbol=symbol, granularity=granularity, params=params, orch=orch)
    path = resolve_dl_model_path(dl_config, symbol)
    save_model_checkpoint(
        path,
        model,
        runtime["norm_stats"],
        candle_epoch_value,
        lookback=params["lookback"],
        calibrator=runtime["calibrator"],
        arch=params["arch"],
        val_accuracy=runtime["val_accuracy"],
        val_brier=runtime["val_brier"],
        val_ece=runtime["val_ece"],
        deploy_ok=False,
        granularity=granularity,
        training_history_bars=int(params.get("training_history_bars") or 0) or None,
        label_horizon_bars=max(1, int(params.get("label_horizon_bars", 1))),
        label_mode=str(params.get("label_mode", "spot_forward")),
        label_call_frac=runtime["label_call_frac"],
        pred_call_frac=runtime["pred_call_frac"],
        minority_recall=runtime["minority_recall"],
    )
    runtime["checkpoint_loaded"] = True
    runtime["session_trained"] = True
    runtime["export_ok"] = True
    runtime["checkpoint_preserved"] = False
    clear_force_retrain(orch, symbol)
    reset_bars_since_train(orch, symbol)
    live = live_signal_snapshot(orch, symbol) if orch is not None else {"live_wr": 0.0, "live_n": 0}
    logger.log(
        level,
        "DL TREINO | %s | concluido em %.0fs | epocas=%d | loss=%.4f | val_acc=%.2f | "
        "brier=%.3f | ece=%.3f | label_call=%.2f | pred_call=%.2f | minority_rec=%.2f | "
        "checkpoint tecnico com teto de 1%% | live_wr=%.2f | live_n=%d",
        symbol,
        time.monotonic() - started,
        int(getattr(train_result, "epochs_ran", 0)),
        float(train_result.avg_loss or 0.0),
        float(runtime["val_accuracy"]),
        float(runtime["val_brier"]),
        float(runtime["val_ece"]),
        float(runtime["label_call_frac"]),
        float(runtime["pred_call_frac"]),
        float(runtime["minority_recall"]),
        float(live.get("live_wr", 0.0)),
        int(live.get("live_n", 0)),
    )
    return runtime["norm_stats"], train_result.avg_loss
