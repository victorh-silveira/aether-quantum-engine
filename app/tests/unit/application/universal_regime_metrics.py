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
            "hurst": 0.52,
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
