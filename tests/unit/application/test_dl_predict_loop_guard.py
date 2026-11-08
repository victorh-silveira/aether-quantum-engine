"""Contratos de predicao DL sem asyncio.run no event loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.application.services.deep_learning.dl_predict import predict_symbol_decision


def test_predict_symbol_decision_no_loop_uses_sync():
    orch = MagicMock()
    runtime = {"val_accuracy": 0.6, "lookback": 30}
    params = {"lookback": 30, "val_acc_live_blend": 0.0}
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_symbol_decision_sync",
        return_value={"ok": True},
    ) as sync_fn:
        out = predict_symbol_decision(
            orch,
            "1HZ75V",
            MagicMock(),
            [1.0] * 40,
            None,
            runtime,
            params,
            None,
        )
    assert out == {"ok": True}
    sync_fn.assert_called_once()


@pytest.mark.asyncio
async def test_predict_symbol_decision_under_loop_requires_force_or_async():
    orch = MagicMock()
    runtime = {"val_accuracy": 0.6, "lookback": 30}
    params = {"lookback": 30, "val_acc_live_blend": 0.0}
    with pytest.raises(RuntimeError, match="event loop"):
        predict_symbol_decision(
            orch,
            "1HZ75V",
            MagicMock(),
            [1.0] * 40,
            None,
            runtime,
            params,
            None,
        )
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_symbol_decision_sync",
        return_value={"forced": True},
    ):
        out = predict_symbol_decision(
            orch,
            "1HZ75V",
            MagicMock(),
            [1.0] * 40,
            None,
            runtime,
            params,
            None,
            force_local=True,
        )
    assert out == {"forced": True}
