from unittest.mock import MagicMock, patch

from src.application.services.llm.llm_cluster_propagate import propagate_cluster_decisions
from src.application.services.llm.llm_cluster_propagate_region import _merge_apply_result, cluster_region_active
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


def test_merge_apply_result_keeps_invert_tag_when_propagated():
    propagated: list[str] = []
    blocked: list[str] = []
    inverted: list[str] = []
    corrected: list[str] = []
    assert _merge_apply_result(
        propagated,
        blocked,
        inverted,
        corrected,
        "R_25[P]",
        None,
        "R_25[C->P]",
        None,
    )
    assert propagated == ["R_25[P]"]
    assert inverted == ["R_25[C->P]"]


def test_cluster_region_active_respects_exclusive_macro():
    assert cluster_region_active(exclusive=False, active_region="us", region="eu") is True
    assert cluster_region_active(exclusive=True, active_region="us", region="us") is True
    assert cluster_region_active(exclusive=True, active_region="us", region="eu") is False


def test_propagate_tries_fallback_index_when_leader_blocked():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_25", "R_50"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["R_25", "R_50"], "eu": []},
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
            "metrics": {"execute": sym == "R_50"},
        }
        if sym == "R_25":
            return None, f"{sym}[C]:statarb_z_misaligned", None, None
        return f"{sym}[C]", None, None, None

    with patch(
        "src.application.services.llm.llm_cluster_propagate_region.apply_cluster_target_decision",
        side_effect=side_effect,
    ):
        propagate_cluster_decisions(
            orch,
            anchor_sym="R_100",
            direction=TradeDirection.CALL,
            metrics=_base_metrics(
                us_cluster="CALL",
                eu_cluster="CALL",
                statarb_spreads={"R_25": -2.0, "R_50": 0.2},
                hmm_state=0,
            ),
            decisions=decisions,
            cid="C0099",
        )
    assert "R_50" in decisions
    assert decisions["R_50"]["metrics"]["execute"] is True


def test_propagate_fallback_can_invert_on_alternate_index():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_25", "R_50"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["R_25", "R_50"], "eu": []},
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
        if sym == "R_25":
            return None, f"{sym}[C]:statarb_z_misaligned", None, None
        return None, None, f"{sym}[C->P]", None

    with patch(
        "src.application.services.llm.llm_cluster_propagate_region.apply_cluster_target_decision",
        side_effect=side_effect,
    ):
        propagate_cluster_decisions(
            orch,
            anchor_sym="R_100",
            direction=TradeDirection.CALL,
            metrics=_base_metrics(
                us_cluster="CALL",
                eu_cluster="CALL",
                statarb_spreads={"R_25": -2.0, "R_50": 0.2},
                hmm_state=0,
            ),
            decisions=decisions,
            cid="C0100",
        )
    assert any("CLUSTER_INVERT" in str(c) for c in orch.logger.info.call_args_list)


def test_propagate_tries_fallback_when_leader_repeat_loss_with_invert():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_50", "R_25"]
    orch._last_loss_symbol = "R_50"
    orch._last_loss_direction = "PUT"
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["R_50", "R_25"], "eu": []},
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": True,
                "best_symbol_only": True,
                "cluster_invert_llm_side": True,
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
            "metrics": {"execute": sym == "R_25"},
        }
        if sym == "R_50":
            return None, f"{sym}[P]:repeat_loss_setup", f"{sym}[C->P]", None
        return f"{sym}[P]", None, f"{sym}[C->P]", None

    with patch(
        "src.application.services.llm.llm_cluster_propagate_region.apply_cluster_target_decision",
        side_effect=side_effect,
    ):
        propagate_cluster_decisions(
            orch,
            anchor_sym="R_100",
            direction=TradeDirection.CALL,
            metrics=_base_metrics(
                macro_sentiment="divergence_us_leads",
                macro_us_strength_quant=0.70,
                macro_eu_strength_quant=0.30,
                us_cluster="CALL",
                eu_cluster="PUT",
                statarb_spreads={"R_50": -0.76, "R_25": 0.5},
                hmm_state=0,
            ),
            decisions=decisions,
            cid="C0103",
        )
    assert decisions["R_25"]["metrics"]["execute"] is True
    assert decisions["R_50"]["metrics"]["execute"] is False


def test_propagate_fallback_records_blocked_alternate():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_25", "R_50"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["R_25", "R_50"], "eu": []},
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
        if sym == "R_25":
            return None, f"{sym}[C]:statarb_z_misaligned", None, None
        return None, f"{sym}[C]:low_conviction", None, None

    with patch(
        "src.application.services.llm.llm_cluster_propagate_region.apply_cluster_target_decision",
        side_effect=side_effect,
    ):
        propagate_cluster_decisions(
            orch,
            anchor_sym="R_100",
            direction=TradeDirection.CALL,
            metrics=_base_metrics(
                us_cluster="CALL",
                eu_cluster="CALL",
                statarb_spreads={"R_25": -2.0, "R_50": 0.2},
                hmm_state=0,
            ),
            decisions=decisions,
            cid="C0101",
        )
    assert "R_25" in decisions and "R_50" in decisions
    assert any("CLUSTER_BLOCK" in str(c) for c in orch.logger.info.call_args_list)


def test_propagate_fallback_returns_on_corrected_alternate():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_25", "R_50"]
    orch.config = {
        "llm": {"min_conviction_execute": 0.60},
        "strategy": {
            "clusters": {"us": ["R_25", "R_50"], "eu": []},
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
        if sym == "R_25":
            return None, f"{sym}[C]:statarb_z_misaligned", None, None
        return "R_50[P]", None, None, "R_50[C->P]"

    with patch(
        "src.application.services.llm.llm_cluster_propagate_region.apply_cluster_target_decision",
        side_effect=side_effect,
    ):
        propagate_cluster_decisions(
            orch,
            anchor_sym="R_100",
            direction=TradeDirection.CALL,
            metrics=_base_metrics(
                us_cluster="CALL",
                eu_cluster="CALL",
                statarb_spreads={"R_25": -2.0, "R_50": 0.2},
                hmm_state=0,
            ),
            decisions=decisions,
            cid="C0102",
        )
    assert any("CLUSTER_BEST" in str(c) or "CLUSTER_PROP" in str(c) for c in orch.logger.info.call_args_list)
