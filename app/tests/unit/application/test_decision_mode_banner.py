from unittest.mock import MagicMock

from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner
from src.infrastructure.handlers.stream_timeframe import granularity_label


def test_emit_decision_engine_banner_dl_enabled():
    logger = MagicMock()
    emit_decision_engine_banner(
        logger,
        {
            "deep_learning": {
                "arch": "tcn",
                "lookback": 720,
                "train_timeframe": "micro",
                "confidence_call_threshold": 0.62,
                "confidence_put_threshold": 0.38,
                "online_training": False,
            },
            "data_handler": {"granularity": 300, "micro_granularity": 60},
            "risk_management": {"params": {"duration": 30, "duration_unit": "s"}},
            "orchestrator": {"execution": {"mandatory_trade_each_cycle": False}},
        },
        decision_mode="deep_learning",
    )
    logger.info.assert_called_once()
    fmt = logger.info.call_args.args[0]
    args = logger.info.call_args.args[1:]
    rendered = fmt % args
    assert "ohlc=60s (micro)" in rendered
    assert "macro=300s" in rendered
    assert "micro=60s" in rendered
    assert "contrato=30s" in rendered
    assert "lb=720" in rendered
    assert "continuo" in rendered


def test_emit_decision_engine_banner_macro_train_timeframe():
    logger = MagicMock()
    emit_decision_engine_banner(
        logger,
        {
            "deep_learning": {"arch": "tcn", "lookback": 72, "train_timeframe": "macro"},
            "data_handler": {"granularity": 300, "micro_granularity": 60},
            "risk_management": {"params": {"duration": 30, "duration_unit": "s"}},
        },
        decision_mode="deep_learning",
    )
    rendered = logger.info.call_args.args[0] % logger.info.call_args.args[1:]
    assert "ohlc=300s (macro)" in rendered


def test_emit_decision_engine_banner_inactive():
    logger = MagicMock()
    emit_decision_engine_banner(logger, {}, decision_mode="inactive")
    logger.debug.assert_called_once()
    assert "INATIVO" in logger.debug.call_args.args[0]


def test_granularity_label_minutes_and_hours():
    assert granularity_label(60) == "M1"
    assert granularity_label(120) == "M2"
    assert granularity_label(300) == "M5"
    assert granularity_label(600) == "M10"
    assert granularity_label(3600) == "H1"
    assert granularity_label(86400) == "D1"
    assert granularity_label(30) == "S30"
