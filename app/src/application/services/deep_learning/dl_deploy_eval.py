"""Mini walk-forward de deploy sem import circular com dl_params."""

from typing import Any

from src.application.services.deep_learning.dl_deploy import call_target_label, direction_wins
from src.application.services.deep_learning.dl_gate_config import deploy_params_for_eval, parse_deploy_gate_config
from src.application.services.deep_learning.dl_labels import LabelSpec
from src.application.services.deep_learning.dl_predict import predict_symbol_decision


def _deploy_eval_bar_indices(start: int, end: int, max_steps: int) -> list[int]:
    """Retorna indices de barras para mini walk-forward respeitando teto de passos."""
    indices = list(range(start, end))
    cap = max(1, int(max_steps))
    if len(indices) <= cap:
        return indices
    step = max(1, (len(indices) + cap - 1) // cap)
    return indices[::step][:cap]


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
        return True, float(runtime.get("val_accuracy", 0.5)), float(runtime.get("val_brier", 1.0))
    lookback = int(runtime.get("lookback", params["lookback"]))
    mini = max(lookback + 5, int(cfg["mini_bars"]))
    if len(prices) < mini + 2:
        return False, 0.0, float(runtime.get("val_brier", 1.0))
    start = len(prices) - mini
    wins = 0
    total = 0
    brier_acc = 0.0
    eval_params = deploy_params_for_eval(params, cfg)
    sim_runtime = dict(runtime)
    sim_runtime["deploy_ok"] = True
    max_steps = int(cfg.get("max_eval_steps", 24))
    label_spec = LabelSpec.from_dl_config(params)
    bars = _deploy_eval_bar_indices(start, len(prices) - 1, max_steps)
    for bar in bars:
        window = prices[: bar + 1]
        win_open = open_[: bar + 1] if open_ is not None else None
        win_high = high[: bar + 1] if high is not None else None
        win_low = low[: bar + 1] if low is not None else None
        win_micro = None
        if micro:
            win_micro = {k: v[: bar + 1] for k, v in micro.items()}
        entry = predict_symbol_decision(
            orch,
            symbol,
            model,
            window,
            norm_stats,
            sim_runtime,
            eval_params,
            None,
            recovery_active=False,
            open_=win_open,
            high=win_high,
            low=win_low,
            micro=win_micro,
            force_local=True,
        )
        if not entry["metrics"].get("execute") or entry["direction"] is None:
            continue
        direction = entry["direction"]
        won = direction_wins(direction, prices, bar, label_spec=label_spec)
        total += 1
        if won:
            wins += 1
        score = float(entry["metrics"].get("raw_prob", 0.5))
        label = call_target_label(prices, bar, label_spec=label_spec)
        brier_acc += (score - label) ** 2
    if total < int(cfg["min_trades"]):
        return False, 0.0, float(runtime.get("val_brier", 1.0))
    win_rate = wins / float(total)
    val_brier = brier_acc / float(total)
    deploy_ok = val_brier + 1e-9 < float(cfg["max_brier"]) and win_rate + 1e-9 >= float(cfg["min_win_rate"])
    return deploy_ok, win_rate, val_brier
