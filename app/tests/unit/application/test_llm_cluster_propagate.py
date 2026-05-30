from unittest.mock import MagicMock

from src.application.services.llm.llm_cluster_propagate import propagate_cluster_decisions
from src.domain.models.trade import TradeDirection


def test_propagate_skips_when_exclusive_macro_tie():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_25", "R_50"]
    orch.config = {
        "strategy": {
            "correlation": {"exclusive_cluster_by_macro": True},
            "clusters": {"us": ["R_25", "R_50"], "eu": []},
        },
    }
    decisions: dict = {}
    propagate_cluster_decisions(
        orch,
        anchor_sym="R_100",
        direction=TradeDirection.CALL,
        metrics={
            "macro_sentiment": "indefinido",
            "macro_us_strength_quant": 0.5,
            "macro_eu_strength_quant": 0.5,
            "us_cluster": "CALL",
            "eu_cluster": "CALL",
        },
        decisions=decisions,
        cid="t1",
    )
    assert decisions == {}
    orch.logger.debug.assert_called()
