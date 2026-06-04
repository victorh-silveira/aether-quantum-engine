from unittest.mock import MagicMock

from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner


def test_emit_decision_engine_banner_dl_enabled():
    logger = MagicMock()
    emit_decision_engine_banner(
        logger,
        {"deep_learning": {"model_path": "data/x.pth", "lookback": 20}},
        dl_enabled=True,
    )
    logger.info.assert_called_once()
    assert "DEEP_LEARNING_PYTORCH" in logger.info.call_args.args[0]


def test_emit_decision_engine_banner_dl_disabled():
    logger = MagicMock()
    emit_decision_engine_banner(logger, {}, dl_enabled=False)
    logger.debug.assert_called_once()
    assert "INATIVO" in logger.debug.call_args.args[0]
