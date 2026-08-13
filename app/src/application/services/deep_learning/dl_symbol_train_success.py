"""Persistencia de checkpoint e metricas apos treino walk-forward."""

import logging
import time
from pathlib import Path

import numpy as np

from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_deploy import apply_deploy_to_runtime
from src.application.services.deep_learning.dl_deploy_eval import evaluate_mini_deploy
from src.application.services.deep_learning.dl_gate_config import (
    describe_deploy_block,
    resolve_deploy_ok,
)
from src.application.services.deep_learning.dl_horizon import contract_duration_seconds
from src.application.services.deep_learning.dl_model_artifacts import schedule_model_upload
from src.application.services.deep_learning.dl_retrain import clear_force_retrain, reset_bars_since_train
from src.application.services.deep_learning.dl_sharpness import (
    assert_export_sharpness_value,
    resolve_calibration_sharpness_cfg,
)
from src.application.services.deep_learning.dl_symbol_runtime import resolve_dl_model_path
from src.application.services.deep_learning.model import save_model_checkpoint
from src.application.services.live_signal_metrics import live_signal_snapshot


logger = logging.getLogger("AETH")


def _log_horizon_gap(
    *,
    level: int,
    symbol: str,
    granularity: int,
    params: dict,
    orch,
) -> None:
    """Loga gap entre horizonte de label e duracao do contrato."""
    label_horizon_bars = max(1, int(params.get("label_horizon_bars", 1)))
    label_horizon_seconds = int(label_horizon_bars) * max(1, int(granularity))
    risk_cfg = getattr(orch, "config", {}) if orch is not None else {}
    risk = risk_cfg.get("risk_management") if isinstance(risk_cfg, dict) else {}
    risk_params = risk.get("params") if isinstance(risk, dict) else {}
    if not isinstance(risk_params, dict):
        risk_params = params.get("risk_params") if isinstance(params.get("risk_params"), dict) else {}
    contract_sec = (
        contract_duration_seconds(risk_params) if risk_params else int(params.get("contract_duration_seconds", 0) or 0)
    )
    logger.log(
        level,
        "DL TREINO | %s | horizonte label=%ds (%d barras x %ds) | contrato=%ds | gap=%ds",
        symbol,
        label_horizon_seconds,
        label_horizon_bars,
        int(granularity),
        int(contract_sec),
        int(label_horizon_seconds - contract_sec),
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
    gate_cfg: dict,
    candle_epoch_value: int,
    granularity: int,
    level: int,
    started: float,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro=None,
) -> tuple[object, float]:
    """Persiste checkpoint, deploy gate e metricas apos treino walk-forward valido."""
    runtime["norm_stats"] = train_result.norm_stats
    norm_stats = train_result.norm_stats
    runtime["val_accuracy"] = train_result.val_accuracy
    runtime["calibrator"] = train_result.calibrator or CalibratorState()
    runtime["val_brier"] = train_result.val_brier
    runtime["val_ece"] = train_result.val_ece
    runtime["calibrated_entropy"] = float(getattr(train_result, "calibrated_entropy", 0.0))
    runtime["entropy_violation"] = bool(getattr(train_result, "entropy_violation", False))
    train_loss = train_result.avg_loss
    runtime["last_candle_epoch"] = candle_epoch_value
    _log_horizon_gap(level=level, symbol=symbol, granularity=granularity, params=params, orch=orch)
    mini_ok, deploy_wr, mini_brier = evaluate_mini_deploy(
        orch,
        symbol,
        model,
        prices,
        norm_stats,
        runtime,
        params,
        gate_cfg=gate_cfg,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
    )
    deploy_ok = resolve_deploy_ok(
        mini_ok=mini_ok,
        val_accuracy=float(train_result.val_accuracy),
        val_brier=float(train_result.val_brier),
        gate_cfg=gate_cfg,
        label_call_frac=float(getattr(train_result, "label_call_frac", 0.5)),
        pred_call_frac=float(getattr(train_result, "pred_call_frac", 0.5)),
        minority_recall=float(getattr(train_result, "minority_recall", 1.0)),
    )
    apply_deploy_to_runtime(
        runtime,
        deploy_ok=deploy_ok,
        deploy_win_rate=deploy_wr,
        val_brier=mini_brier if mini_ok else float(train_result.val_brier),
    )
    calib_cfg = dl_config.get("calibration") if isinstance(dl_config, dict) else None
    sharpness_cfg = resolve_calibration_sharpness_cfg(calib_cfg if isinstance(calib_cfg, dict) else None)
    oos_sharpness = float(getattr(train_result, "oos_sharpness", 0.0))
    raw_sharpness = float(getattr(train_result, "raw_sharpness", 0.0))
    try:
        assert_export_sharpness_value(
            oos_sharpness,
            floor=float(sharpness_cfg["min_oos_sharpness"]),
            label="holdout",
        )
    except RuntimeError as exc:
        logger.warning(
            "DL TREINO | %s | sharpness abaixo do piso — deploy_ok=false | %s (raw_sharpness=%.4f method=%s)",
            symbol,
            exc,
            raw_sharpness,
            getattr(runtime.get("calibrator"), "method", "?"),
        )
        runtime["deploy_ok"] = False
    runtime["oos_sharpness"] = oos_sharpness
    runtime["raw_sharpness"] = raw_sharpness
    runtime["label_call_frac"] = float(getattr(train_result, "label_call_frac", 0.5))
    runtime["pred_call_frac"] = float(getattr(train_result, "pred_call_frac", 0.5))
    runtime["minority_recall"] = float(getattr(train_result, "minority_recall", 1.0))
    path = Path(resolve_dl_model_path(dl_config, symbol))
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
        deploy_settlement_win_rate=float(runtime.get("deploy_settlement_win_rate", 0.0)),
        deploy_settlement_brier=float(runtime.get("deploy_settlement_brier", runtime.get("val_brier", 1.0))),
        deploy_settlement_n=int(runtime.get("deploy_settlement_n", 0) or 0),
        oos_sharpness=float(runtime.get("oos_sharpness", 0.0)),
        granularity=granularity,
        training_history_bars=int(params.get("training_history_bars") or 0) or None,
        label_call_frac=float(runtime.get("label_call_frac", 0.5)),
        pred_call_frac=float(runtime.get("pred_call_frac", 0.5)),
        minority_recall=float(runtime.get("minority_recall", 1.0)),
    )
    schedule_model_upload(
        orch,
        symbol,
        path,
        arch=str(params["arch"]),
        metadata={
            "val_accuracy": runtime["val_accuracy"],
            "calibrated_entropy": runtime.get("calibrated_entropy"),
            "entropy_violation": runtime.get("entropy_violation"),
        },
    )
    runtime["session_trained"] = bool(runtime.get("deploy_ok", False))
    runtime["export_ok"] = bool(runtime.get("deploy_ok", False))
    runtime["checkpoint_preserved"] = False
    clear_force_retrain(orch, symbol)
    reset_bars_since_train(orch, symbol)
    live_snap = live_signal_snapshot(orch, symbol) if orch is not None else {"live_wr": 0.0, "live_n": 0}
    live_wr = float(live_snap.get("live_wr", 0.0))
    live_n = int(live_snap.get("live_n", 0))
    logger.log(
        level,
        "DL TREINO | %s | concluido em %.0fs | epocas=%d | loss=%.4f | val_acc=%.2f | brier=%.3f | "
        "deploy=%s | settle_wr=%.2f | settle_brier=%.3f | label_wr=%.2f | live_wr=%.2f | live_n=%d | "
        "label_call=%.2f | pred_call=%.2f | minority_rec=%.2f",
        symbol,
        time.monotonic() - started,
        int(getattr(train_result, "epochs_ran", 0)),
        float(train_loss or 0.0),
        float(runtime.get("val_accuracy", 0.0)),
        float(runtime.get("val_brier", 1.0)),
        bool(runtime.get("deploy_ok", False)),
        float(runtime.get("deploy_settlement_win_rate", deploy_wr)),
        float(runtime.get("deploy_settlement_brier", mini_brier)),
        float(runtime.get("deploy_label_win_rate", deploy_wr)),
        live_wr,
        live_n,
        float(runtime.get("label_call_frac", 0.5)),
        float(runtime.get("pred_call_frac", 0.5)),
        float(runtime.get("minority_recall", 1.0)),
    )
    if not bool(runtime.get("deploy_ok", False)):
        reason = describe_deploy_block(
            mini_ok=bool(mini_ok),
            val_accuracy=float(train_result.val_accuracy),
            val_brier=float(train_result.val_brier),
            gate_cfg=gate_cfg,
            label_call_frac=float(getattr(train_result, "label_call_frac", 0.5)),
            pred_call_frac=float(getattr(train_result, "pred_call_frac", 0.5)),
            minority_recall=float(getattr(train_result, "minority_recall", 1.0)),
        )
        logger.warning(
            "DL TREINO | %s | deploy_ok=false (%s) — retreinar; nao iniciar meta",
            symbol,
            reason,
        )
    logger.log(level, "")
    return norm_stats, train_loss
