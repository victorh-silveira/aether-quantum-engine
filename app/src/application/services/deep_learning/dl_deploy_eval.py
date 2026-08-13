"""Mini walk-forward de deploy sem import circular com dl_params."""

from typing import Any

from src.application.services.deep_learning.dl_deploy import call_target_label, direction_wins
from src.application.services.deep_learning.dl_gate_config import deploy_params_for_eval, parse_deploy_gate_config
from src.application.services.deep_learning.dl_horizon import contract_duration_seconds
from src.application.services.deep_learning.dl_labels import LABEL_MODE_SPOT, LabelSpec
from src.application.services.deep_learning.dl_predict import predict_symbol_decision


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
    label_spec: LabelSpec,
    settlement_spec: LabelSpec,
) -> tuple[bool, bool, float, float] | None:
    """Avalia um bar de deploy e retorna (ok, win, conf, settlement)."""
    window = prices[: bar + 1]
    win_open = open_[: bar + 1] if open_ is not None else None
    win_high = high[: bar + 1] if high is not None else None
    win_low = low[: bar + 1] if low is not None else None
    win_micro = {k: v[: bar + 1] for k, v in micro.items()} if micro else None
    entry = predict_symbol_decision(
        orch,
        symbol,
        model,
        window,
        norm_stats,
        sim_runtime,
        eval_params,
        None,
        open_=win_open,
        high=win_high,
        low=win_low,
        micro=win_micro,
        force_local=True,
    )
    if not entry["metrics"].get("execute") or entry["direction"] is None:
        return None
    direction = entry["direction"]
    won = direction_wins(direction, prices, bar, label_spec=label_spec)
    settlement_won = direction_wins(direction, prices, bar, label_spec=settlement_spec)
    score = float(entry["metrics"].get("raw_prob", 0.5))
    label = call_target_label(prices, bar, label_spec=label_spec)
    settlement_label = call_target_label(prices, bar, label_spec=settlement_spec)
    return won, settlement_won, (score - label) ** 2, (score - settlement_label) ** 2


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
    if not cfg.get("enabled", True):
        runtime["deploy_settlement_n"] = 0
        return True, float(runtime.get("val_accuracy", 0.5)), float(runtime.get("val_brier", 1.0))
    lookback = int(runtime.get("lookback", params["lookback"]))
    mini = max(lookback + 5, int(cfg["mini_bars"]))
    if len(prices) < mini + 2:
        runtime["deploy_settlement_n"] = 0
        return False, 0.0, float(runtime.get("val_brier", 1.0))
    start = len(prices) - mini
    wins = 0
    settlement_wins = 0
    total = 0
    brier_acc = 0.0
    settlement_brier_acc = 0.0
    eval_params = deploy_params_for_eval(params, cfg)
    sim_runtime = dict(runtime)
    sim_runtime["deploy_ok"] = True
    max_steps = int(cfg.get("max_eval_steps", 24))
    label_spec = LabelSpec.from_dl_config(params)
    gran = int(params.get("granularity") or runtime.get("granularity") or 3600)
    settlement_horizon = resolve_settlement_horizon_bars(params, gran)
    settlement_spec = LabelSpec(
        horizon_bars=settlement_horizon,
        smooth_bars=1,
        label_mode=LABEL_MODE_SPOT,
        ma_window=label_spec.ma_window,
    )
    for bar in _deploy_eval_bar_indices(start, len(prices) - 1, max_steps):
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
            label_spec=label_spec,
            settlement_spec=settlement_spec,
        )
        if scored is None:
            continue
        won, settlement_won, brier_term, settlement_brier_term = scored
        total += 1
        wins += int(won)
        settlement_wins += int(settlement_won)
        brier_acc += brier_term
        settlement_brier_acc += settlement_brier_term
    if total < int(cfg["min_trades"]):
        runtime["deploy_settlement_win_rate"] = 0.0
        runtime["deploy_settlement_brier"] = float(runtime.get("val_brier", 1.0))
        runtime["deploy_label_win_rate"] = 0.0
        runtime["deploy_settlement_n"] = int(total)
        return False, 0.0, float(runtime.get("val_brier", 1.0))
    win_rate = wins / float(total)
    val_brier = brier_acc / float(total)
    settlement_wr = settlement_wins / float(total)
    settlement_brier = settlement_brier_acc / float(total)
    runtime["deploy_label_win_rate"] = float(win_rate)
    runtime["deploy_settlement_win_rate"] = float(settlement_wr)
    runtime["deploy_settlement_brier"] = float(settlement_brier)
    runtime["deploy_settlement_horizon_bars"] = int(settlement_horizon)
    runtime["deploy_settlement_n"] = int(total)
    min_wr = float(cfg["min_win_rate"])
    max_brier = float(cfg["max_brier"])
    deploy_ok = settlement_brier + 1e-9 < max_brier and settlement_wr + 1e-9 >= min_wr
    if not deploy_ok:
        deploy_ok = val_brier + 1e-9 < max_brier and win_rate + 1e-9 >= min_wr
    return deploy_ok, settlement_wr if deploy_ok else win_rate, settlement_brier if deploy_ok else val_brier
