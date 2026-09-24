"""Mini walk-forward de deploy sem import circular com dl_params."""

import logging
import time
from pathlib import Path
from typing import Any

import joblib

from src.application.services.deep_learning.dl_deploy import call_target_label, direction_wins
from src.application.services.deep_learning.dl_gate_config import deploy_params_for_eval, parse_deploy_gate_config
from src.application.services.deep_learning.dl_horizon import contract_duration_seconds
from src.application.services.deep_learning.dl_labels import LABEL_MODE_SPOT, LabelSpec
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_statistical_gate import wilson_lower_bound
from src.application.services.meta_classifier_features import extract_meta_feature_vector


logger = logging.getLogger("AETH")


def _load_active_meta_model(params: dict[str, Any] | None = None) -> Any:
    """Carrega modelo Meta-Learner ativo para filtragem conjunta OOS."""
    meta_path = Path("infra/docker/meta-models/meta_lgbm.pkl")
    if params and isinstance(params.get("meta_model_path"), (str, Path)):
        meta_path = Path(params["meta_model_path"])
    if not meta_path.is_file():
        return None
    try:
        payload = joblib.load(meta_path)
        if isinstance(payload, dict):
            return payload.get("model")
        return payload
    except Exception as exc:
        logger.debug("Erro ao carregar meta model: %s", exc)
        return None


def _deploy_eval_bar_indices(start: int, end: int, max_steps: int) -> list[int]:
    """Retorna indices de barras para mini walk-forward respeitando teto de passos."""
    indices = list(range(start, end))
    cap = max(1, int(max_steps))
    if len(indices) <= cap:
        return indices
    step = max(1, (len(indices) + cap - 1) // cap)
    return indices[::step][:cap]


def resolve_settlement_horizon_bars(params: dict[str, Any], granularity: int) -> int:
    """Barras de settlement = duracao do contrato na granularidade de avaliacao."""
    risk = params.get("risk_params") if isinstance(params.get("risk_params"), dict) else {}
    if not risk:
        duration = int(params.get("contract_duration_seconds", 0) or 0)
        if duration <= 0:
            return 1
        return max(1, int(round(duration / float(max(1, granularity)))))
    return max(1, int(round(contract_duration_seconds(risk) / float(max(1, granularity)))))


def _score_deploy_bar(
    *,
    orch,
    symbol: str,
    model,
    prices,
    norm_stats,
    sim_runtime: dict,
    eval_params: dict,
    bar: int,
    open_,
    high,
    low,
    micro,
    settlement_spec: LabelSpec,
    meta_model: Any = None,
) -> tuple[bool, float, float] | None:
    """Avalia um bar de deploy e retorna settlement e Brier operacional com filtro meta."""
    infer_bars = int(eval_params.get("inference_history_bars", bar + 1))
    start = max(0, bar + 1 - max(1, infer_bars))
    window = prices[start : bar + 1]
    win_open = open_[start : bar + 1] if open_ is not None else None
    win_high = high[start : bar + 1] if high is not None else None
    win_low = low[start : bar + 1] if low is not None else None
    win_micro = {k: v[start : bar + 1] for k, v in micro.items()} if micro else None
    entry = predict_symbol_decision(
        orch,
        symbol,
        model,
        window,
        norm_stats,
        sim_runtime,
        eval_params,
        None,
        granularity=int(eval_params.get("granularity") or sim_runtime.get("trained_granularity") or 60),
        open_=win_open,
        high=win_high,
        low=win_low,
        micro=win_micro,
        force_local=True,
    )
    if not entry["metrics"].get("execute") or entry["direction"] is None:
        return None
    if meta_model is not None:
        vector = extract_meta_feature_vector(entry["metrics"])
        try:
            pred_edge = float(meta_model.predict([vector])[0])
            min_edge = float(eval_params.get("meta_min_edge", 0.0))
            if pred_edge < min_edge:
                return None
        except Exception as exc:
            logger.debug("Falha na inferencia do meta_model: %s", exc)
    direction = entry["direction"]
    settlement_won = direction_wins(direction, prices, bar, label_spec=settlement_spec)
    score = float(entry["metrics"].get("raw_prob", 0.5))
    settlement_label = call_target_label(prices, bar, label_spec=settlement_spec)
    return settlement_won, (score - settlement_label) ** 2, score


def evaluate_mini_deploy(
    orch,
    symbol: str,
    model,
    prices,
    norm_stats,
    runtime: dict,
    params: dict[str, Any],
    *,
    gate_cfg: dict[str, Any] | None = None,
    open_=None,
    high=None,
    low=None,
    micro=None,
) -> tuple[bool, float, float]:
    """Simula ultimas barras com gating atual e retorna deploy_ok, win_rate, brier."""
    cfg = gate_cfg or parse_deploy_gate_config(params if "deploy_gate" in params else {})
    runtime["deploy_settlement_source"] = "m5_close_proxy"
    if not cfg.get("enabled", True):
        runtime["deploy_settlement_n"] = 0
        return True, float(runtime.get("val_accuracy", 0.5)), float(runtime.get("val_brier", 1.0))
    lookback = int(runtime.get("lookback", params["lookback"]))
    mini = max(lookback + 5, int(cfg["mini_bars"]))
    if len(prices) < mini + 2:
        runtime["deploy_settlement_n"] = 0
        return False, 0.0, float(runtime.get("val_brier", 1.0))
    start = len(prices) - mini
    settlement_wins = 0
    total = 0
    settlement_brier_acc = 0.0
    eval_params = deploy_params_for_eval(params, cfg)
    sim_runtime = dict(runtime)
    sim_runtime["deploy_ok"] = True
    max_steps = int(cfg.get("max_eval_steps", 24))
    enforce_settle = bool(cfg.get("enforce_settle_gate", True))
    if max_steps <= 0 or int(cfg.get("mini_bars", 1)) <= 0:
        val_acc = float(runtime.get("val_accuracy", 0.5))
        val_brier = float(runtime.get("val_brier", 0.25))
        gran = int(params.get("granularity") or runtime.get("granularity") or 3600)
        runtime["deploy_settlement_n"] = 0
        runtime["deploy_settlement_win_rate"] = val_acc
        runtime["deploy_label_win_rate"] = val_acc
        runtime["deploy_settlement_brier"] = val_brier
        runtime["deploy_settlement_wilson_lcb"] = val_acc
        runtime["deploy_settlement_horizon_bars"] = resolve_settlement_horizon_bars(params, gran)
        deploy_ok = not enforce_settle or bool(cfg.get("force_ok", False))
        runtime["deploy_provisional_ok"] = deploy_ok
        logger.info(
            "SETTLE | Avaliacao OOS ignorada (max_eval_steps=%d, mini_bars=%d, enforce_settle=%s)",
            max_steps,
            int(cfg.get("mini_bars", 1)),
            enforce_settle,
        )
        return deploy_ok, val_acc, val_brier
    source_spec = LabelSpec.from_dl_config(params)
    gran = int(params.get("granularity") or runtime.get("granularity") or 3600)
    settlement_horizon = resolve_settlement_horizon_bars(params, gran)
    settlement_spec = LabelSpec(
        horizon_bars=settlement_horizon,
        smooth_bars=1,
        label_mode=LABEL_MODE_SPOT,
        ma_window=source_spec.ma_window,
    )
    meta_model = _load_active_meta_model(params)
    runtime["deploy_meta_filter_applied"] = bool(meta_model is not None)
    eval_bars = _deploy_eval_bar_indices(start, len(prices) - 1, max_steps)
    total_eval_steps = len(eval_bars)
    log_interval_steps = max(1, total_eval_steps // 10)
    t_start = time.perf_counter()
    logger.info(
        "SETTLE | Iniciando avaliacao OOS | passos=%d | intervalo_log=%d",
        total_eval_steps,
        log_interval_steps,
    )
    for step, bar in enumerate(eval_bars, 1):
        scored = _score_deploy_bar(
            orch=orch,
            symbol=symbol,
            model=model,
            prices=prices,
            norm_stats=norm_stats,
            sim_runtime=sim_runtime,
            eval_params=eval_params,
            bar=bar,
            open_=open_,
            high=high,
            low=low,
            micro=micro,
            settlement_spec=settlement_spec,
            meta_model=meta_model,
        )
        if scored is not None:
            settlement_won, settlement_brier_term, _ = scored
            total += 1
            settlement_wins += int(settlement_won)
            settlement_brier_acc += settlement_brier_term
        if step % log_interval_steps == 0 or step == total_eval_steps:
            elapsed = time.perf_counter() - t_start
            pct = (step / total_eval_steps) * 100.0 if total_eval_steps > 0 else 100.0
            eta = (elapsed / step) * (total_eval_steps - step) if step > 0 else 0.0
            current_wr = (settlement_wins / total) if total > 0 else 0.0
            logger.info(
                "SETTLE | progresso=%d/%d (%.1f%%) | trades=%d | partial_wr=%.2f%% | tempo=%.0fs | eta=%.0fs",
                step,
                total_eval_steps,
                pct,
                total,
                current_wr * 100.0,
                elapsed,
                eta,
            )
    if total < int(cfg["min_trades"]):
        runtime["deploy_settlement_win_rate"] = 0.0
        runtime["deploy_settlement_brier"] = float(runtime.get("val_brier", 1.0))
        runtime["deploy_label_win_rate"] = 0.0
        runtime["deploy_settlement_n"] = int(total)
        if not enforce_settle:
            runtime["deploy_provisional_ok"] = True
            return True, float(runtime.get("val_accuracy", 0.5)), float(runtime.get("val_brier", 0.25))
        return False, 0.0, float(runtime.get("val_brier", 1.0))
    settlement_wr = settlement_wins / float(total)
    settlement_brier = settlement_brier_acc / float(total)
    runtime["deploy_label_win_rate"] = float(settlement_wr)
    runtime["deploy_settlement_win_rate"] = float(settlement_wr)
    runtime["deploy_settlement_brier"] = float(settlement_brier)
    runtime["deploy_settlement_horizon_bars"] = int(settlement_horizon)
    runtime["deploy_settlement_n"] = int(total)
    settlement_lcb = wilson_lower_bound(
        wins=settlement_wins,
        trials=total,
        confidence=float(cfg.get("settlement_confidence", 0.90)),
    )
    runtime["deploy_settlement_wilson_lcb"] = float(settlement_lcb)
    min_wr = float(cfg["min_win_rate"])
    max_brier = float(cfg["max_brier"])
    deploy_ok = settlement_brier + 1e-9 < max_brier and settlement_lcb + 1e-9 >= min_wr
    if bool(cfg.get("require_broker_settlement", False)):
        deploy_ok = False
    runtime["deploy_provisional_ok"] = bool(
        total >= int(cfg.get("provisional_min_trades", 48))
        and settlement_brier + 1e-9 < float(cfg.get("provisional_max_brier", max_brier))
        and settlement_wr + 1e-9 >= float(cfg.get("provisional_min_win_rate", min_wr))
    )
    if bool(cfg.get("require_broker_settlement", False)):
        runtime["deploy_provisional_ok"] = False
    if not enforce_settle:
        deploy_ok = True
        runtime["deploy_provisional_ok"] = True
    return deploy_ok, settlement_wr, settlement_brier
