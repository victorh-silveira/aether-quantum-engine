TREND_EXPANSION_INDICATORS = {
    "adx": 0.28,
    "hurst": 0.58,
    "vol_ratio": 1.10,
    "rsi": 0.55,
    "cmo": 0.10,
    "bb_width": 0.05,
}


def asymmetric_gate_safe_metrics(**overrides):
    defaults = {
        "deploy_ok": True,
        "execute": True,
        "val_accuracy": 0.60,
        "edge": 0.08,
        "trade_score": 0.75,
        "conviction": 0.75,
        "raw_prob": 0.75,
        "indicators": dict(TREND_EXPANSION_INDICATORS),
    }
    defaults.update(overrides)
    return base_metrics(**defaults)


def base_metrics(**overrides):
    metrics = {
        "trade_score": 0.82,
        "conviction": 0.82,
        "resolved_conviction": 0.82,
        "direction_call_score": 0.72,
        "direction_put_score": 0.28,
        "dl_direction": "CALL",
        "exec_direction": "CALL",
        "call_votes": 4,
        "put_votes": 2,
        "raw_prob": 0.72,
        "calibrated_prob": 0.72,
        "indicators": {
            "vol_ratio": 1.0,
            "adx": 0.25,
            "hurst": 0.58,
            "rsi": 0.52,
            "cmo": 0.10,
            "keltner": 0.50,
            "bb_width": 0.05,
        },
    }
    metrics.update(overrides)
    if "indicators" in overrides:
        merged = dict(metrics["indicators"])
        merged.update(overrides["indicators"])
        metrics["indicators"] = merged
    return metrics


def bear_put_metrics(**overrides):
    defaults = {
        "trade_score": 0.72,
        "conviction": 0.72,
        "direction_call_score": 0.28,
        "direction_put_score": 0.72,
        "dl_direction": "PUT",
        "exec_direction": "PUT",
        "raw_prob": 0.28,
        "calibrated_prob": 0.28,
        "cross_symbol_features": {"cross_symbol_vol_ratio_diff": -0.1},
    }
    defaults.update(overrides)
    return asymmetric_gate_safe_metrics(**defaults)
