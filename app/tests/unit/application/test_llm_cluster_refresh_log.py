from src.application.services.llm.llm_cluster_refresh_log import (
    _dir_label,
    effective_cluster_refresh_line,
)
from src.domain.models.trade import TradeDirection


def test_dir_label_invalid_entry():
    assert _dir_label(None) == "-"
    assert _dir_label({"direction": "CALL"}) == "-"


def test_effective_cluster_refresh_line_shows_eff_and_cache():
    decisions = {
        "OTC_DJI": {
            "direction": TradeDirection.PUT,
            "metrics": {"cluster_active_region": "us"},
        },
        "OTC_FCHI": {
            "direction": TradeDirection.CALL,
            "metrics": {"cluster_active_region": "eu"},
        },
        "frxEURUSD": {
            "direction": TradeDirection.PUT,
            "metrics": {"cluster_active_region": "us"},
        },
    }
    metrics = {"us_cluster": "CALL", "eu_cluster": "PUT"}
    line = effective_cluster_refresh_line(
        decisions,
        anchor_sym="frxEURUSD",
        metrics=metrics,
        macro_tag="divergence_us_leads",
    )
    assert "us_eff=PUT" in line
    assert "eu_eff=CALL" in line
    assert "llm_cache=CALL/PUT" in line
    assert "macro=divergence_us_leads" in line


def test_effective_cluster_refresh_line_skips_non_dict_entry():
    line = effective_cluster_refresh_line(
        {"OTC_DJI": "bad"},
        anchor_sym="frxEURUSD",
        metrics={},
        macro_tag="risk_off",
    )
    assert "us_eff=-" in line
