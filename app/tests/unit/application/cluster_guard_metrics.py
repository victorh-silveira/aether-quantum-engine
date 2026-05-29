def base_cluster_metrics(**overrides):
    base = {
        "conviction": 0.70,
        "execute": False,
        "us_cluster": "PUT",
        "eu_cluster": "PUT",
        "macro_sentiment": "risk_off",
        "macro_us_strength_quant": 0.30,
        "macro_eu_strength_quant": 0.72,
        "hmm_prob": 0.90,
    }
    base.update(overrides)
    return base
