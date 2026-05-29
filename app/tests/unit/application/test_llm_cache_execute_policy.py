from src.application.services.llm.llm_refresh_policy import clear_cluster_execute_on_cached_decisions
from src.domain.models.trade import TradeDirection


def test_clear_cluster_execute_on_cached_decisions():
    cached = {
        "frxEURUSD": {"direction": TradeDirection.CALL, "metrics": {"execute": True}},
        "OTC_DJI": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "llm_block_reason": "allowed"},
        },
    }
    out = clear_cluster_execute_on_cached_decisions(cached, "frxEURUSD")
    assert out["frxEURUSD"]["metrics"]["execute"] is True
    assert out["OTC_DJI"]["metrics"]["execute"] is False
    assert out["OTC_DJI"]["metrics"]["llm_block_reason"] == "cache_no_fresh_llm"


def test_clear_cluster_execute_skips_non_dict_entries():
    out = clear_cluster_execute_on_cached_decisions({"OTC_DJI": "bad"}, "frxEURUSD")
    assert out == {}
