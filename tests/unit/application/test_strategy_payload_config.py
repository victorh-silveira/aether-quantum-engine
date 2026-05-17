from src.application.services.llm import IndicatorConfig
from src.application.services.llm.sniper_payload import build_sniper_tokens, format_sniper_prompt_line
from src.application.services.llm.strategy_payload_config import resolve_strategy_payload_config


def test_resolve_strategy_maps_hurst_labels_to_payload():
    root = {
        "strategy": {
            "thresholds": {
                "hurst": {"persist": "TREND", "anti": "REV", "random": "NOISE", "na": "NA"},
            },
            "payload": {"token_order": ["hurst", "zscore", "entropy", "velocity"]},
        }
    }
    sp = resolve_strategy_payload_config(root)
    ic = IndicatorConfig(hurst_window=30)
    closes = [100.0 * (1.01**i) for i in range(50)]
    tok = build_sniper_tokens(closes, ic, sp)
    assert tok["hurst"] == "TREND"


def test_format_sniper_prompt_line_respects_field_labels():
    root = {
        "strategy": {
            "payload": {
                "field_labels": {"hurst": "H", "zscore": "Z", "entropy": "E", "velocity": "V"},
                "pair_separator": " | ",
                "kv_separator": ":",
            }
        }
    }
    sp = resolve_strategy_payload_config(root)
    line = format_sniper_prompt_line(
        "X",
        "Momentum Alpha (Bull)",
        "random_walk",
        "Mean Reversion Alpha",
        "HIGH_ENTROPY_REGIME",
        {"hurst": "p", "zscore": "h", "entropy": "l", "velocity": "u"},
        sp,
        mtf_alignment_line="M30: momentum | M15: random | M5: mean | M1: noise",
    )
    assert "H:p" in line
    assert " | " in line
    assert "MTF:P/N/M/N" in line
