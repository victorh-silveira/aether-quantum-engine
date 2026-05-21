from unittest.mock import MagicMock

import numpy as np

from src.application.services.llm.global_macro_confluence import empty_macro_snapshot
from src.application.services.llm.indicators import IndicatorConfig
from src.application.services.llm.symbol_decision_post import (
    append_entropy_high_note,
    apply_conviction_inversion,
    cluster_index_directions_for_orch,
    patch_final_symbol_metrics,
)
from src.domain.models.trade import TradeDirection


def test_append_entropy_high_note_when_threshold_met():
    swing = list(np.linspace(1.0, 2.0, 40))
    runtime = {"indicator_config": IndicatorConfig()}
    out = append_entropy_high_note("base", 0.9, swing, runtime, None)
    assert "ENTROPY_HIGH" in out


def test_apply_conviction_inversion_and_cluster_directions():
    d, note, inv = apply_conviction_inversion(TradeDirection.CALL, 0.4, "n", {"inversion_threshold": 0.5})
    assert inv is True
    assert d == TradeDirection.PUT
    d2, note2, inv2 = apply_conviction_inversion(
        TradeDirection.CALL, 0.52, "n", {"inversion_threshold": 0.5, "follow_threshold": 0.6}
    )
    assert inv2 is False
    assert "Follow" in note2
    orch = MagicMock()
    orch.config = {
        "strategy": {
            "clusters": {"us": ["OTC_SPC"], "eu": ["OTC_FCHI"]},
            "correlation": {"index_direction_mode": "counter_trend"},
        }
    }
    snap = empty_macro_snapshot()
    dirs = cluster_index_directions_for_orch(orch, snap)
    assert isinstance(dirs, dict)
    metrics: dict = {}
    patch_final_symbol_metrics(
        metrics,
        execute_flag=True,
        inverted=False,
        llm_http_ms=1.0,
        llm_resp_chars=10,
        llm_direction_from_api=True,
        us_dir=TradeDirection.PUT,
        eu_dir=None,
        macro_snapshot=snap,
        macro_guard=False,
        cluster_index_directions=dirs,
    )
    assert metrics["execute"] is True
    assert metrics["us_cluster"] == "PUT"
