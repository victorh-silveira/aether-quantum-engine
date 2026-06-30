from unittest.mock import MagicMock

from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner


def test_emit_decision_engine_banner_dl_enabled():
    logger = MagicMock()
    emit_decision_engine_banner(
        logger,
        {
            "deep_learning": {
                "arch": "tcn",
                "lookback": 48,
                "confidence_call_threshold": 0.75,
                "confidence_put_threshold": 0.25,
            },
            "risk_management": {"params": {"duration": 60, "duration_unit": "s"}},
            "orchestrator": {"execution": {"mandatory_trade_each_cycle": False}},
        },
        decision_mode="deep_learning",
    )
    logger.info.assert_called_once()
    assert "continuo" in logger.info.call_args.args[0]


def test_emit_decision_engine_banner_inactive():
    logger = MagicMock()
    emit_decision_engine_banner(logger, {}, decision_mode="inactive")
    logger.debug.assert_called_once()
    assert "INATIVO" in logger.debug.call_args.args[0]
