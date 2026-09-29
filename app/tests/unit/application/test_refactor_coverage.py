"""Cobertura dos contratos de qualidade e reconciliacao passiva."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.deep_learning.dl_sharpness import sharpness_pass_fraction
from src.application.services.orchestrator.orchestrator_run_loop import _run_passive_settlement_reconcile


def test_sharpness_pass_fraction_empty_and_threshold():
    assert sharpness_pass_fraction([], floor=0.1) == 0.0
    assert sharpness_pass_fraction([0.5, 0.7], floor=0.1) == 0.5


@pytest.mark.asyncio
async def test_passive_reconcile_logs_cleared_state():
    orch = SimpleNamespace(logger=MagicMock())
    with patch("src.application.services.orchestrator.orchestrator_run_loop.SettlementOrphanCleaner") as cleaner_class:
        cleaner_class.return_value.passive_reconcile = AsyncMock(return_value=True)
        await _run_passive_settlement_reconcile(orch)
    orch.logger.info.assert_called_once()
