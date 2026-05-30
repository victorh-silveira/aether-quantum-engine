from src.application.services.llm.llm_refresh_policy import clear_cluster_execute_on_cached_decisions
from src.domain.models.trade import TradeDirection


def test_clear_cluster_execute_on_cached_decisions():
    cached = {
        "R_100": {"direction": TradeDirection.CALL, "metrics": {"execute": True}},
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "llm_block_reason": "allowed"},
        },
    }
    out = clear_cluster_execute_on_cached_decisions(cached, "R_100")
    assert out["R_100"]["metrics"]["execute"] is True
    assert out["R_50"]["metrics"]["execute"] is False
    assert out["R_50"]["metrics"]["llm_block_reason"] == "cache_no_fresh_llm"


def test_clear_cluster_execute_skips_non_dict_entries():
    out = clear_cluster_execute_on_cached_decisions({"R_50": "bad"}, "R_100")
    assert out == {}
