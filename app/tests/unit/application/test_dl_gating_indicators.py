from src.application.services.deep_learning.dl_params import parse_indicator_gating_config


def test_parse_indicator_gating_config_defaults_when_missing():
    cfg = parse_indicator_gating_config({})
    assert cfg["enabled"] is False
    assert cfg["veto_on_noise"] is False
    assert cfg["hurst_min"] == 0.0


def test_parse_indicator_gating_config_ignores_override_bounds():
    cfg = parse_indicator_gating_config(
        {
            "indicator_gating": {
                "enabled": True,
                "hurst_min": 0.43,
                "hurst_max": 0.95,
                "veto_on_noise": True,
                "noise_hurst_lo": 0.45,
                "noise_hurst_hi": 0.55,
            }
        }
    )
    assert cfg["enabled"] is False
    assert cfg["veto_on_noise"] is False
