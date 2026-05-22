from src.application.services.llm.llm_bridge_guards import (
    merge_mtf_scores,
    mtf_score,
    mtf_score_from_alignment,
)
from src.application.services.llm.llm_bridge_telemetry import (
    attach_decision_telemetry,
    build_metrics_for_decision,
    format_llm_runtime_thresholds,
    store_symbol_decision,
)
from src.application.services.llm.llm_bridge_utils import (
    canonical_direction_token,
    extract_think_block,
    parse_llm_trade_response,
    score_token,
    trend_token,
)
from src.domain.models.trade import TradeDirection


def test_canonical_direction_token_coverage():
    assert canonical_direction_token("CALL") == "CALL"
    assert canonical_direction_token("PUT") == "PUT"
    assert canonical_direction_token("WAIT") is None
    assert canonical_direction_token("SKIP") is None
    assert canonical_direction_token("HOLD") is None
    assert canonical_direction_token("NEUTRAL") is None
    assert canonical_direction_token("AGUARDAR") is None
    assert canonical_direction_token("UNKNOWN") is None
    assert canonical_direction_token("") is None


def test_parsing_utils_coverage():
    assert extract_think_block("no think") == ("", "no think")
    assert extract_think_block("<think>abc</think>def") == ("abc", "def")

    assert parse_llm_trade_response("CALL and PUT")["note"] == "CALL_AMBIGUOUS"
    assert parse_llm_trade_response("JUST WAIT")["direction"] is None
    assert parse_llm_trade_response("55% prob")["conviction"] == 0.55
    assert parse_llm_trade_response("<think>abc</think>CALL")["_think"] == "abc"


def test_trend_and_score_utils():
    assert trend_token("comprador") == "alta"
    assert trend_token("vendedor") == "baixa"
    assert trend_token("nada") is None
    assert score_token("bullish") == 1
    assert score_token("bearish") == -1
    assert score_token("none") == 0


def test_mtf_score_logic():
    assert mtf_score("alta", "alta", "baixa", "baixa") == (1 + 1 - 2 - 3)
    assert mtf_score_from_alignment("alta x alta x baixa x baixa") == (1 + 1 - 2 - 3)
    assert mtf_score_from_alignment("alta x alta x baixa") == (1 + 2 - 3)
    assert mtf_score_from_alignment("too short") == 0

    assert merge_mtf_scores(1, 0, "m15: alta | m5: alta | m3: alta") == 0
    assert merge_mtf_scores(1, 0, "simple") == 1
    assert merge_mtf_scores(1, 2) == 2


def test_stubs_and_log_formatters():
    assert (
        format_llm_runtime_thresholds({"timeout": 10, "num_predict": 100, "min_conviction_execute": 0.5})
        == "tout=10.0s toks=100 mconv=0.50"
    )


def test_telemetry_and_storage():
    class MockOrch:
        _active_cycle_id = 1

    metrics = {}
    attach_decision_telemetry(metrics, {}, "range", 0.5, "src", MockOrch(), "OTC_FCHI")
    assert metrics["regime_label"] == "range"

    decisions = {}
    store_symbol_decision(decisions, "OTC_FCHI", TradeDirection.CALL, {"m": 1})
    assert decisions["OTC_FCHI"]["direction"] == TradeDirection.CALL


def test_metrics_finalization():
    runtime = {"model": "test"}
    _, m = build_metrics_for_decision(
        runtime, TradeDirection.CALL, 0.8, "n", 1.0, "a", "b", "c", "d", "e", lambda d, c, n: {"d": d}
    )
    assert m["execute"] is True
