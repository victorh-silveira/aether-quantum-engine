from src.application.services.execution_direction_cross_corr import adjust_dl_weight_with_correlation


def test_cross_corr_reduces_dl_weight_on_divergent_high_correlation():
    weights = {"dl_raw_weight": 0.45}
    metrics = {
        "direction_margin": 0.02,
        "indicators": {"vol_ratio": 0.7, "di_diff": 0.2},
        "bb_squeeze": True,
    }
    corr = {("1HZ75V", "1HZ75V"): 0.8}
    adjusted = adjust_dl_weight_with_correlation(weights, "1HZ75V", metrics, corr, min_margin=0.05)
    assert adjusted["dl_raw_weight"] < weights["dl_raw_weight"]


def test_cross_corr_inactive_when_strong_consensus():
    weights = {"dl_raw_weight": 0.45}
    metrics = {
        "direction_margin": 0.2,
        "indicators": {"vol_ratio": 1.1, "di_diff": 0.0},
    }
    corr = {("R_10", "R_50"): 0.9, ("R_50", "R_10"): 0.9}
    adjusted = adjust_dl_weight_with_correlation(weights, "R_10", metrics, corr, min_margin=0.05)
    assert adjusted["dl_raw_weight"] == weights["dl_raw_weight"]
