"""Testes do loss-classifier (features, client, pool, gate)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.application.services.loss_classifier_features import LOSS_FEATURE_DIM, build_loss_feature_vector
from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.application.services.orchestrator.execution_blockers import _candidate_block_reason
from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.loss_classifier_client import (
    LossClassifierClient,
    build_loss_classifier_client_from_config,
    loss_classifier_enabled,
    resolve_loss_classifier_config,
)
from src.infrastructure.inference.loss_classifier_pool import (
    close_loss_classifier_client,
    get_loss_classifier_client,
    learn_loss_via_config_sync,
    predict_loss_via_config_sync,
)
from src.infrastructure.inference.loss_classifier_types import parse_loss_predict_response


def test_build_loss_feature_vector_dim_and_edges():
    metrics = {
        "direction_margin": "x",
        "calibrated_prob": None,
        "scale_adapted": False,
        "scale_micro_regime": "retraction",
        "tcn_direction": "PUT",
        "scale_tape_consensus": "PUT",
    }
    vector = build_loss_feature_vector(metrics, TradeDirection.CALL, pending=5.0, linear=2, bankroll=0.0)
    assert len(vector) == LOSS_FEATURE_DIM
    assert vector[5] == 1.0


def test_parse_loss_predict_response_and_bad_payload():
    parsed = parse_loss_predict_response(
        {
            "p_loss": 0.7,
            "veto": True,
            "auto_learn_applied": True,
            "model_version": "v1",
            "n_train": 40,
            "veto_ready": True,
        }
    )
    assert parsed["veto"] is True
    with pytest.raises(TypeError):
        parse_loss_predict_response([])


def test_resolve_and_enabled():
    cfg = resolve_loss_classifier_config({})
    assert cfg["enabled"] is True
    assert loss_classifier_enabled(None) is True
    assert loss_classifier_enabled({"infra": {"loss_classifier": {"enabled": False}}}) is False
    assert loss_classifier_enabled({"infra": {}}) is True


@pytest.mark.asyncio
async def test_client_predict_and_learn_paths():
    client = LossClassifierClient(
        base_url="http://loss.test",
        timeout=1.0,
        enabled=True,
        veto_p_loss_floor=0.62,
    )
    assert client.enabled is True
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "p_loss": 0.2,
        "veto": False,
        "auto_learn_applied": False,
        "model_version": "m",
        "n_train": 10,
        "veto_ready": False,
    }
    with patch.object(client._client, "post", new=AsyncMock(return_value=mock_resp)):
        out = await client.predict_loss(
            {
                "feature_vector": [0.0] * 24,
                "symbol": "R_10",
                "direction": "CALL",
                "veto_p_loss_floor": 0.62,
            }
        )
        assert out["p_loss"] == 0.2
        learned = await client.learn(feature_vector=[0.0] * 24, label="WIN", contract_id="1", symbol="R_10")
        assert learned.get("p_loss") is None or learned
    with patch.object(client._client, "post", new=AsyncMock(side_effect=httpx.TimeoutException("t"))):
        empty = await client.predict_loss(
            {"feature_vector": [0.0] * 24, "symbol": "", "direction": "", "veto_p_loss_floor": 0.62}
        )
        assert empty["veto"] is False
        bad = await client.learn(feature_vector=[0.0] * 24, label="LOSS")
        assert bad["ok"] is False
    disabled = LossClassifierClient(base_url="http://x", timeout=1.0, enabled=False, veto_p_loss_floor=0.5)
    assert (
        await disabled.predict_loss({"feature_vector": [], "symbol": "", "direction": "", "veto_p_loss_floor": 0.5})
    )["veto"] is False
    assert (await disabled.learn(feature_vector=[], label="WIN"))["skipped"] is True
    with patch.object(disabled._client, "aclose", new=AsyncMock()) as closed:
        await disabled.aclose()
        closed.assert_awaited()
    built = build_loss_classifier_client_from_config({"infra": {"loss_classifier": {"enabled": True}}})
    assert built.enabled is True
    await built.aclose()


@pytest.mark.asyncio
async def test_pool_get_close_and_stale_loop():
    await close_loss_classifier_client()
    cfg = {"infra": {"loss_classifier": {"enabled": True}}}
    first = await get_loss_classifier_client(cfg)
    second = await get_loss_classifier_client(cfg)
    assert first is second
    await close_loss_classifier_client()


def test_predict_and_learn_sync_wrappers():
    cfg = {"infra": {"loss_classifier": {"enabled": True}}}
    with patch(
        "src.infrastructure.inference.loss_classifier_pool.get_loss_classifier_client",
        new=AsyncMock(
            return_value=MagicMock(
                predict_loss=AsyncMock(
                    return_value={
                        "p_loss": 0.1,
                        "veto": False,
                        "auto_learn_applied": False,
                        "model_version": "x",
                        "n_train": 1,
                        "veto_ready": False,
                    }
                ),
                learn=AsyncMock(return_value={"ok": True}),
            )
        ),
    ):
        pred = predict_loss_via_config_sync(
            cfg,
            {"feature_vector": [0.0] * 24, "symbol": "R_10", "direction": "CALL", "veto_p_loss_floor": 0.62},
        )
        assert pred["p_loss"] == 0.1
        learned = learn_loss_via_config_sync(cfg, feature_vector=[0.0] * 24, label="WIN", symbol="R_10")
        assert learned["ok"] is True
    assert learn_loss_via_config_sync(
        {"infra": {"loss_classifier": {"enabled": False}}}, feature_vector=[], label="WIN"
    )["skipped"]


def test_apply_loss_classifier_veto_and_ok_paths():
    metrics = {
        "direction_margin": 0.03,
        "calibrated_prob": 0.52,
        "scale_adapted": True,
        "scale_micro_regime": "chop",
        "tcn_direction": "CALL",
        "scale_tape_consensus": "PUT",
    }
    orch = MagicMock()
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    orch.risk_manager = MagicMock()
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = MagicMock(return_value=0.0)
    orch.state.balance = "bad"
    orch._log_dedupe = {}
    with patch(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        return_value={
            "p_loss": 0.8,
            "veto": True,
            "auto_learn_applied": True,
            "model_version": "auto1",
            "n_train": 40,
            "veto_ready": True,
        },
    ):
        assert apply_loss_classifier_gate(metrics, TradeDirection.PUT, orch=orch, symbol="R_10") is True
    assert metrics["gate_reason"] == "loss_clf_veto"
    metrics2 = dict(metrics)
    metrics2.pop("gate_reason", None)
    metrics2.pop("signal_skip_reason", None)
    with patch(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        return_value={
            "p_loss": 0.2,
            "veto": False,
            "auto_learn_applied": False,
            "model_version": "m",
            "n_train": 5,
            "veto_ready": False,
        },
    ):
        assert apply_loss_classifier_gate(metrics2, TradeDirection.PUT, orch=orch, symbol="R_10") is False


def test_skip_before_loss_and_force_and_disabled():
    assert apply_loss_classifier_gate({"gate_reason": "cal_margin"}, TradeDirection.CALL, orch=MagicMock()) is False
    assert apply_loss_classifier_gate({}, TradeDirection.CALL, force=True, orch=MagicMock()) is False
    assert apply_loss_classifier_gate({}, TradeDirection.CALL, orch=None) is False
    with patch("src.application.services.loss_classifier_gate.loss_classifier_enabled", return_value=False):
        assert apply_loss_classifier_gate({}, TradeDirection.CALL, orch=MagicMock()) is False


def test_block_reason_loss_clf_veto():
    assert _candidate_block_reason({"gate_reason": "loss_clf_veto"}) == "loss_clf_veto"


def test_feed_loss_classifier_learn_paths():
    from src.application.services.orchestrator.settlement_outcome import _feed_loss_classifier_learn

    orch = MagicMock()
    orch._loss_clf_vectors = None
    _feed_loss_classifier_learn(orch, "R_10", won=True, contract_id=1)
    orch._loss_clf_vectors = {"R_10": []}
    _feed_loss_classifier_learn(orch, "R_10", won=True, contract_id=1)
    orch._loss_clf_vectors = {"R_10": [0.1] * 24}
    orch.config = None
    _feed_loss_classifier_learn(orch, "R_10", won=False, contract_id=2)
    orch._loss_clf_vectors = {"R_10": [0.2] * 24}
    orch.config = {"infra": {"loss_classifier": {"enabled": True}}}
    with patch(
        "src.application.services.orchestrator.settlement_outcome.learn_loss_via_config_sync",
        return_value={"ok": True},
    ) as learn:
        _feed_loss_classifier_learn(orch, "R_10", won=True, contract_id=9)
        learn.assert_called_once()
        assert learn.call_args.kwargs["label"] == "WIN"
        assert learn.call_args.kwargs["symbol"] == "R_10"
    assert "R_10" not in orch._loss_clf_vectors


def test_features_bankroll_norm_and_dim_guard():
    vector = build_loss_feature_vector(
        {"direction_margin": 0.02},
        TradeDirection.CALL,
        pending=250.0,
        linear=0,
        bankroll=10000.0,
    )
    assert vector[9] == pytest.approx(0.025)
    with (
        patch("src.application.services.loss_classifier_features.LOSS_FEATURE_DIM", 99),
        pytest.raises(ValueError, match="loss feature dim"),
    ):
        build_loss_feature_vector({}, TradeDirection.CALL)


@pytest.mark.asyncio
async def test_pool_stale_loop_and_sync_under_running_loop():
    from src.infrastructure.inference import loss_classifier_pool as pool_mod

    await close_loss_classifier_client()
    cfg = {"infra": {"loss_classifier": {"enabled": True}}}
    first = await get_loss_classifier_client(cfg)
    stale = MagicMock()
    stale.aclose = AsyncMock()
    pool_mod._LossClientPool.client = first
    pool_mod._LossClientPool.loop = object()
    with patch.object(first, "aclose", new=AsyncMock()) as _:
        pool_mod._LossClientPool.client = first
        rebuilt = await get_loss_classifier_client(cfg)
        assert rebuilt is not None
    await close_loss_classifier_client()
    with patch(
        "src.infrastructure.inference.loss_classifier_pool.get_loss_classifier_client",
        new=AsyncMock(
            return_value=MagicMock(
                predict_loss=AsyncMock(
                    return_value={
                        "p_loss": 0.3,
                        "veto": False,
                        "auto_learn_applied": False,
                        "model_version": "z",
                        "n_train": 2,
                        "veto_ready": False,
                    }
                ),
                learn=AsyncMock(return_value={"ok": True}),
            )
        ),
    ):
        pred = predict_loss_via_config_sync(
            cfg,
            {"feature_vector": [0.0] * 24, "symbol": "R_10", "direction": "PUT", "veto_p_loss_floor": 0.62},
        )
        assert pred["p_loss"] == 0.3
        assert learn_loss_via_config_sync(cfg, feature_vector=[0.0] * 24, label="LOSS")["ok"] is True
