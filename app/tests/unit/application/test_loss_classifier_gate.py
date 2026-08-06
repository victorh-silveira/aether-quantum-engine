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
    }
    vector = build_loss_feature_vector(metrics, TradeDirection.CALL, pending=5.0, linear=2, bankroll=0.0)
    assert len(vector) == LOSS_FEATURE_DIM
    assert vector[5] == 1.0
    assert vector[23] == 0.0
    oppose = build_loss_feature_vector(
        {**metrics, "scale_mini_prev_bar_dir": "PUT", "scale_mini_bar_dir": "PUT"},
        TradeDirection.CALL,
        pending=5.0,
        linear=2,
        bankroll=1000.0,
    )
    assert oppose[23] == 1.0


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
    assert cfg["veto_mode"] == "soft"
    assert cfg["veto_p_loss_floor"] == pytest.approx(0.65)
    assert cfg["soft_kelly_mult"] == pytest.approx(0.55)
    assert cfg["soft_kelly_mult_high"] == pytest.approx(0.40)
    assert cfg["soft_p_loss_high"] == pytest.approx(0.85)
    assert cfg["soft_max_stake_pct_high"] == pytest.approx(0.02)
    assert cfg["timeout_seconds"] == pytest.approx(8.0)
    assert cfg["retrain_on_loss_min_n"] == 2
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
    with pytest.raises(ValueError, match="soft_p_loss_high"):
        resolve_loss_classifier_config({"soft_p_loss_high": 0.50})


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
                "symbol": "OTC_SPC",
                "direction": "CALL",
                "veto_p_loss_floor": 0.62,
            }
        )
        assert out["p_loss"] == 0.2
        learned = await client.learn(feature_vector=[0.0] * 24, label="WIN", contract_id="1", symbol="OTC_SPC")
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
            {"feature_vector": [0.0] * 24, "symbol": "OTC_SPC", "direction": "CALL", "veto_p_loss_floor": 0.62},
        )
        assert pred["p_loss"] == 0.1
        learned = learn_loss_via_config_sync(cfg, feature_vector=[0.0] * 24, label="WIN", symbol="OTC_SPC")
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
            {"feature_vector": [0.0] * 24, "symbol": "OTC_SPC", "direction": "PUT", "veto_p_loss_floor": 0.62},
        )
        assert pred["p_loss"] == 0.3
        assert learn_loss_via_config_sync(cfg, feature_vector=[0.0] * 24, label="LOSS")["ok"] is True


def test_should_retrain_after_learn_loss_forces_when_ready():
    import importlib.util
    from pathlib import Path

    policy_path = Path(__file__).resolve().parents[4] / "infra" / "docker" / "loss-classifier" / "learn_policy.py"
    spec = importlib.util.spec_from_file_location("loss_learn_policy", policy_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.should_retrain_after_learn(label="LOSS", buffer_n=2, retrain_min_n=24, retrain_on_loss_min_n=2) is True
    assert mod.should_retrain_after_learn(label="LOSS", buffer_n=1, retrain_min_n=24, retrain_on_loss_min_n=2) is False
    assert mod.should_retrain_after_learn(label="WIN", buffer_n=2, retrain_min_n=24, retrain_on_loss_min_n=2) is False
    assert mod.should_retrain_after_learn(label="WIN", buffer_n=24, retrain_min_n=24, retrain_on_loss_min_n=2) is True
    assert mod.should_retrain_after_learn(label="WIN", buffer_n=25, retrain_min_n=24, retrain_on_loss_min_n=2) is False
    assert mod.should_retrain_after_learn(label="WIN", buffer_n=32, retrain_min_n=24, retrain_on_loss_min_n=2) is True
    assert mod.retrain_min_for_label(label="LOSS", retrain_min_n=24, retrain_on_loss_min_n=2) == 2
    assert mod.retrain_min_for_label(label="WIN", retrain_min_n=24, retrain_on_loss_min_n=2) == 24


def test_learn_buffer_io_roundtrip(tmp_path):
    import importlib.util
    from pathlib import Path

    io_path = Path(__file__).resolve().parents[4] / "infra" / "docker" / "loss-classifier" / "buffer_io.py"
    spec = importlib.util.spec_from_file_location("loss_buffer_io", io_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    x_rows = [[0.1] * 24, [0.2] * 24]
    y_rows = [0, 1]
    mod.save_learn_buffer(tmp_path, x_rows, y_rows)
    loaded = mod.load_learn_buffer(tmp_path)
    assert loaded is not None
    assert loaded[0] == x_rows
    assert loaded[1] == y_rows
    assert mod.buffer_class_counts(y_rows) == {"win": 1, "loss": 1, "n": 2}
