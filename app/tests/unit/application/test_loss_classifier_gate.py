"""Testes do loss-classifier (features, client, pool, gate)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.application.services.loss_classifier_features import LOSS_FEATURE_DIM, build_loss_feature_vector
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
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "predicted_payoff_edge": 9.5,
        "flow_features": {"micro_tick_acceleration": 4.2},
    }
    vector = build_loss_feature_vector(metrics, TradeDirection.CALL, pending=5.0, linear=2, bankroll=0.0)
    assert len(vector) == LOSS_FEATURE_DIM
    assert vector[5] == 1.0
    assert vector[10] == pytest.approx(3.0)
    assert vector[19] == pytest.approx(3.0)
    assert vector[23] == 0.0
    oppose = build_loss_feature_vector(
        {**metrics, "scale_mini_prev_bar_dir": "PUT", "scale_mini_bar_dir": "PUT"},
        TradeDirection.CALL,
        pending=5.0,
        linear=2,
        bankroll=1000.0,
    )
    assert oppose[23] == 1.0
    neg = build_loss_feature_vector(
        {**metrics, "predicted_payoff_edge": -8.0, "flow_features": {"micro_tick_acceleration": -5.0}},
        TradeDirection.CALL,
    )
    assert neg[10] == pytest.approx(-3.0)
    assert neg[19] == pytest.approx(-3.0)


def test_parse_loss_predict_response_and_bad_payload():
    parsed = parse_loss_predict_response(
        {
            "p_loss": 0.7,
            "veto": True,
            "auto_learn_applied": True,
            "model_version": "v1",
            "n_train": 40,
            "veto_ready": True,
            "collapsed": True,
        }
    )
    assert parsed["veto"] is True
    assert parsed["collapsed"] is True
    with pytest.raises(TypeError):
        parse_loss_predict_response([])


def test_resolve_and_enabled():
    cfg = resolve_loss_classifier_config({})
    assert cfg["enabled"] is True
    assert cfg["veto_mode"] == "soft"
    assert cfg["veto_p_loss_floor"] == pytest.approx(0.65)
    assert cfg["hard_p_loss_floor"] == pytest.approx(0.90)
    assert cfg["hard_blocks_pending_waive"] is True
    assert cfg["soft_kelly_mult"] == pytest.approx(0.55)
    assert cfg["soft_kelly_mult_high"] == pytest.approx(0.40)
    assert cfg["soft_p_loss_high"] == pytest.approx(0.85)
    assert cfg["soft_max_stake_pct_high"] == pytest.approx(0.01)
    assert cfg["timeout_seconds"] == pytest.approx(8.0)
    assert cfg["retrain_on_loss_min_n"] == 1
    assert cfg["retrain_min_n"] == 1
    assert loss_classifier_enabled(None) is True
    assert loss_classifier_enabled({"infra": {"loss_classifier": {"enabled": False}}}) is False
    assert loss_classifier_enabled({"infra": {}}) is True
    with pytest.raises(ValueError, match="veto_mode"):
        resolve_loss_classifier_config({"veto_mode": "block"})
    with pytest.raises(ValueError, match="veto_mode"):
        resolve_loss_classifier_config({"veto_mode": "hard"})
    with pytest.raises(ValueError, match="soft_kelly_mult"):
        resolve_loss_classifier_config({"veto_mode": "soft", "soft_kelly_mult": 0.0})
    with pytest.raises(ValueError, match="soft_kelly_mult_high"):
        resolve_loss_classifier_config({"soft_kelly_mult_high": 0.8})
    with pytest.raises(ValueError, match="hard_p_loss_floor"):
        resolve_loss_classifier_config({"hard_p_loss_floor": 0.50})
    with pytest.raises(ValueError, match="hard_p_loss_floor"):
        resolve_loss_classifier_config({"hard_p_loss_floor": 1.01})
    with pytest.raises(ValueError, match="soft_p_loss_high"):
        resolve_loss_classifier_config({"soft_p_loss_high": 0.50})
    with pytest.raises(ValueError, match="soft_kelly_mult_high"):
        resolve_loss_classifier_config({"soft_kelly_mult_high": 0.0})
    with pytest.raises(ValueError, match="soft_max_stake_pct_high"):
        resolve_loss_classifier_config({"soft_max_stake_pct_high": 0.06})
    with pytest.raises(ValueError, match="flip_cal_discord_margin"):
        resolve_loss_classifier_config({"flip_cal_discord_margin": 0.5})
    with pytest.raises(ValueError, match="flip_min_edge_execute"):
        resolve_loss_classifier_config({"flip_min_edge_execute": 0.9})
    with pytest.raises(ValueError, match="flip_candle_p_loss_floor"):
        resolve_loss_classifier_config({"flip_candle_p_loss_floor": 0.50})
    with pytest.raises(ValueError, match="flip_waive_scale_above_p_loss"):
        resolve_loss_classifier_config({"flip_waive_scale_above_p_loss": 0.80})
    with pytest.raises(ValueError, match="flip_waive_edge_min"):
        resolve_loss_classifier_config({"flip_waive_edge_min": 0.05})


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
