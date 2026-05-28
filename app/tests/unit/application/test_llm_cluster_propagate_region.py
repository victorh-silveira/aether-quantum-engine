from unittest.mock import MagicMock, patch

from src.application.services.llm.llm_cluster_propagate import propagate_cluster_decisions
from src.application.services.llm.llm_cluster_propagate_region import cluster_region_active
from src.domain.models.trade import TradeDirection


def _base_metrics(**overrides):
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


def test_cluster_region_active_respects_exclusive_macro():
    assert cluster_region_active(exclusive=False, active_region="us", region="eu") is True
    assert cluster_region_active(exclusive=True, active_region="us", region="us") is True
    assert cluster_region_active(exclusive=True, active_region="us", region="eu") is False


def test_propagate_tries_fallback_index_when_leader_blocked():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_SPC", "OTC_NDX"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["OTC_SPC", "OTC_NDX"], "eu": []},
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": False,
                "best_symbol_only": True,
                "statarb_try_alternate_on_block": True,
                "statarb_index_select_enabled": True,
                "statarb_index_max_per_cluster": 1,
            },
            "macro": {"statarb_z_threshold": 2.5},
        },
    }
    decisions: dict = {}

    def side_effect(*_args, **kwargs):
        sym = kwargs["target_sym"]
        direction = kwargs["target_direction"]
        kwargs["decisions"][sym] = {
            "direction": direction,
            "metrics": {"execute": sym == "OTC_NDX"},
        }
        if sym == "OTC_SPC":
            return None, f"{sym}[C]:statarb_z_misaligned", None
        return f"{sym}[C]", None, None

    with patch(
        "src.application.services.llm.llm_cluster_propagate_region.apply_cluster_target_decision",
        side_effect=side_effect,
    ):
        propagate_cluster_decisions(
            orch,
            anchor_sym="frxEURUSD",
            direction=TradeDirection.CALL,
            metrics=_base_metrics(
                us_cluster="CALL",
                eu_cluster="CALL",
                statarb_spreads={"OTC_SPC": -2.0, "OTC_NDX": 0.2},
                hmm_state=0,
            ),
            decisions=decisions,
            cid="C0099",
        )
    assert "OTC_NDX" in decisions
    assert decisions["OTC_NDX"]["metrics"]["execute"] is True


def test_propagate_fallback_can_invert_on_alternate_index():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_SPC", "OTC_NDX"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["OTC_SPC", "OTC_NDX"], "eu": []},
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": False,
                "best_symbol_only": True,
                "statarb_try_alternate_on_block": True,
                "statarb_index_select_enabled": True,
                "statarb_index_max_per_cluster": 1,
            },
            "macro": {"statarb_z_threshold": 2.5},
        },
    }
    decisions: dict = {}

    def side_effect(*_args, **kwargs):
        sym = kwargs["target_sym"]
        kwargs["decisions"][sym] = {
            "direction": kwargs["target_direction"],
            "metrics": {"execute": False},
        }
        if sym == "OTC_SPC":
            return None, f"{sym}[C]:statarb_z_misaligned", None
        return None, None, f"{sym}[C->P]"

    with patch(
        "src.application.services.llm.llm_cluster_propagate_region.apply_cluster_target_decision",
        side_effect=side_effect,
    ):
        propagate_cluster_decisions(
            orch,
            anchor_sym="frxEURUSD",
            direction=TradeDirection.CALL,
            metrics=_base_metrics(
                us_cluster="CALL",
                eu_cluster="CALL",
                statarb_spreads={"OTC_SPC": -2.0, "OTC_NDX": 0.2},
                hmm_state=0,
            ),
            decisions=decisions,
            cid="C0100",
        )
    assert any("CLUSTER_INVERT" in str(c) for c in orch.logger.info.call_args_list)


def test_propagate_fallback_records_blocked_alternate():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_SPC", "OTC_NDX"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["OTC_SPC", "OTC_NDX"], "eu": []},
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": False,
                "best_symbol_only": True,
                "statarb_try_alternate_on_block": True,
                "statarb_index_select_enabled": True,
                "statarb_index_max_per_cluster": 1,
            },
            "macro": {"statarb_z_threshold": 2.5},
        },
    }
    decisions: dict = {}

    def side_effect(*_args, **kwargs):
        sym = kwargs["target_sym"]
        kwargs["decisions"][sym] = {"direction": kwargs["target_direction"], "metrics": {"execute": False}}
        if sym == "OTC_SPC":
            return None, f"{sym}[C]:statarb_z_misaligned", None
        return None, f"{sym}[C]:low_conviction", None

    with patch(
        "src.application.services.llm.llm_cluster_propagate_region.apply_cluster_target_decision",
        side_effect=side_effect,
    ):
        propagate_cluster_decisions(
            orch,
            anchor_sym="frxEURUSD",
            direction=TradeDirection.CALL,
            metrics=_base_metrics(
                us_cluster="CALL",
                eu_cluster="CALL",
                statarb_spreads={"OTC_SPC": -2.0, "OTC_NDX": 0.2},
                hmm_state=0,
            ),
            decisions=decisions,
            cid="C0101",
        )
    assert "OTC_SPC" in decisions and "OTC_NDX" in decisions
    assert any("CLUSTER_BLOCK" in str(c) for c in orch.logger.info.call_args_list)
