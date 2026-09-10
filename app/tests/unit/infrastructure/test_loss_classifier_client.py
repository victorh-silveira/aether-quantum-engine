from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.infrastructure.inference.loss_classifier_client import (
    LossClassifierClient,
    build_loss_classifier_client_from_config,
    loss_classifier_enabled,
    resolve_loss_classifier_config,
)


def _loss_cfg(**overrides: object) -> dict:
    base = {
        "enabled": True,
        "http_url": "http://localhost:8006",
        "timeout_seconds": 8.0,
        "max_connections": 8,
        "max_keepalive_connections": 4,
        "feature_dim": 24,
        "veto_mode": "hard",
        "hard_p_loss_floor": 0.55,
        "flip_trust_n": 32,
        "flip_young_shrink": 0.35,
        "flip_young_p_eff_floor": 0.55,
        "ready_n": 32,
        "retrain_min_n": 12,
        "retrain_on_loss_min_n": 4,
        "min_win_for_loss_retrain": 4,
        "max_buffer": 2000,
    }
    base.update(overrides)
    return base


def test_resolve_loss_classifier_config_from_ssot():
    resolved = resolve_loss_classifier_config(None)
    assert resolved["veto_mode"] == "hard"
    assert resolved["hard_p_loss_floor"] == pytest.approx(0.58)
    assert resolved["veto_p_loss_floor"] == pytest.approx(0.58)
    assert resolved["flip_trust_n"] == 32
    assert resolved["flip_young_shrink"] == pytest.approx(0.35)
    assert resolved["flip_young_p_eff_floor"] == pytest.approx(0.55)
    assert resolved["ready_n"] == 32
    assert resolved["retrain_min_n"] == 12
    assert resolved["retrain_on_loss_min_n"] == 4
    assert resolved["min_win_for_loss_retrain"] == 4


def test_resolve_loss_classifier_config_merges_overrides():
    resolved = resolve_loss_classifier_config({"enabled": False, "hard_p_loss_floor": 0.85})
    assert resolved["enabled"] is False
    assert resolved["hard_p_loss_floor"] == pytest.approx(0.85)


def test_resolve_loss_classifier_config_rejects_non_hard_veto():
    with pytest.raises(ValueError, match="veto_mode deve ser hard"):
        resolve_loss_classifier_config({"veto_mode": "soft"})


@pytest.mark.parametrize("floor", [0.0, 1.5])
def test_resolve_loss_classifier_config_rejects_invalid_floor(floor):
    with pytest.raises(ValueError, match="hard_p_loss_floor"):
        resolve_loss_classifier_config({"hard_p_loss_floor": floor})


@pytest.mark.parametrize("shrink", [0.0, 1.5])
def test_resolve_loss_classifier_config_rejects_invalid_young_shrink(shrink):
    with pytest.raises(ValueError, match="flip_young_shrink"):
        resolve_loss_classifier_config({"flip_young_shrink": shrink})


def test_resolve_loss_classifier_config_rejects_invalid_trust_n():
    with pytest.raises(ValueError, match="flip_trust_n"):
        resolve_loss_classifier_config({"flip_trust_n": 0})


@pytest.mark.parametrize("floor", [0.0, 1.5])
def test_resolve_loss_classifier_config_rejects_invalid_young_p_eff_floor(floor):
    with pytest.raises(ValueError, match="flip_young_p_eff_floor"):
        resolve_loss_classifier_config({"flip_young_p_eff_floor": floor})


def test_loss_classifier_enabled_from_root_config():
    assert loss_classifier_enabled({"infra": {"loss_classifier": {"enabled": True}}}) is True
    assert loss_classifier_enabled({"infra": {"loss_classifier": {"enabled": False}}}) is False


def test_loss_classifier_enabled_fallback_when_chunk_missing():
    assert loss_classifier_enabled({}) is bool(resolve_loss_classifier_config(None)["enabled"])
    assert loss_classifier_enabled(None) is bool(resolve_loss_classifier_config(None)["enabled"])


def test_loss_classifier_enabled_without_enabled_key():
    cfg = {"infra": {"loss_classifier": {"http_url": "http://localhost:8006"}}}
    assert loss_classifier_enabled(cfg) is bool(resolve_loss_classifier_config(None)["enabled"])


def test_build_loss_classifier_client_from_config():
    client = build_loss_classifier_client_from_config(
        {"infra": {"loss_classifier": {"enabled": False, "http_url": "http://loss:8006/"}}}
    )
    assert client.enabled is False
    assert client._client.base_url == "http://loss:8006"


def test_build_loss_classifier_client_env_fallback(monkeypatch):
    monkeypatch.delenv("AETHER_LOSS_CLASSIFIER_HTTP", raising=False)
    client = build_loss_classifier_client_from_config(
        {"infra": {"loss_classifier": {"enabled": False, "http_url": ""}}}
    )
    assert str(client._client.base_url).rstrip("/") == "http://localhost:8006"


def test_build_loss_classifier_client_uses_env_when_url_empty(monkeypatch):
    monkeypatch.setenv("AETHER_LOSS_CLASSIFIER_HTTP", "http://env-loss:8006")
    client = build_loss_classifier_client_from_config({"infra": {"loss_classifier": {"enabled": True, "http_url": ""}}})
    assert str(client._client.base_url).rstrip("/") == "http://env-loss:8006"


@pytest.mark.asyncio
async def test_loss_classifier_client_disabled_predict():
    client = LossClassifierClient(base_url="http://loss:8006", timeout=1.0, enabled=False, veto_p_loss_floor=0.9)
    result = await client.predict_loss(
        {"feature_vector": [0.1], "symbol": "1HZ75V", "direction": "CALL", "veto_p_loss_floor": 0.9}
    )
    assert result["p_loss"] == pytest.approx(0.5)
    assert result["veto"] is False
    await client.aclose()


@pytest.mark.asyncio
async def test_loss_classifier_client_predict_success():
    client = LossClassifierClient(base_url="http://loss:8006", timeout=1.0, enabled=True, veto_p_loss_floor=0.9)
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(
        return_value={
            "p_loss": 0.92,
            "veto": True,
            "auto_learn_applied": False,
            "model_version": "v1",
            "n_train": 10,
            "veto_ready": True,
            "bootstrap": False,
            "collapsed": False,
        }
    )
    client._client.post = AsyncMock(return_value=response)
    result = await client.predict_loss(
        {"feature_vector": [0.1, 0.2], "symbol": "1HZ75V", "direction": "PUT", "veto_p_loss_floor": 0.9}
    )
    assert result["p_loss"] == pytest.approx(0.92)
    assert result["veto"] is True
    client._client.post.assert_awaited_once()
    await client.aclose()


@pytest.mark.asyncio
async def test_loss_classifier_client_predict_uses_default_veto_floor():
    client = LossClassifierClient(base_url="http://loss:8006", timeout=1.0, enabled=True, veto_p_loss_floor=0.88)
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"p_loss": 0.5, "veto": False})
    client._client.post = AsyncMock(return_value=response)
    await client.predict_loss({"feature_vector": [0.1], "symbol": "", "direction": "", "veto_p_loss_floor": 0.0})
    payload = client._client.post.await_args.kwargs["json"]
    assert payload["veto_p_loss_floor"] == pytest.approx(0.88)
    await client.aclose()


@pytest.mark.asyncio
async def test_loss_classifier_client_predict_timeout_fallback():
    client = LossClassifierClient(base_url="http://loss:8006", timeout=1.0, enabled=True, veto_p_loss_floor=0.9)
    client._client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    result = await client.predict_loss(
        {"feature_vector": [0.1], "symbol": "1HZ75V", "direction": "CALL", "veto_p_loss_floor": 0.9}
    )
    assert result["p_loss"] == pytest.approx(0.5)
    assert result["veto"] is False
    await client.aclose()


@pytest.mark.asyncio
async def test_loss_classifier_client_predict_http_error_fallback():
    client = LossClassifierClient(base_url="http://loss:8006", timeout=1.0, enabled=True, veto_p_loss_floor=0.9)
    request = httpx.Request("POST", "http://loss:8006/v1/predict_loss")
    response = httpx.Response(503, request=request)
    client._client.post = AsyncMock(side_effect=httpx.HTTPStatusError("fail", request=request, response=response))
    result = await client.predict_loss(
        {"feature_vector": [0.1], "symbol": "1HZ75V", "direction": "CALL", "veto_p_loss_floor": 0.9}
    )
    assert result["veto"] is False
    await client.aclose()


@pytest.mark.asyncio
async def test_loss_classifier_client_predict_invalid_json_fallback():
    client = LossClassifierClient(base_url="http://loss:8006", timeout=1.0, enabled=True, veto_p_loss_floor=0.9)
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=[])
    client._client.post = AsyncMock(return_value=response)
    result = await client.predict_loss(
        {"feature_vector": [0.1], "symbol": "1HZ75V", "direction": "CALL", "veto_p_loss_floor": 0.9}
    )
    assert result["model_version"] == "none"
    await client.aclose()


@pytest.mark.asyncio
async def test_loss_classifier_client_learn_disabled():
    client = LossClassifierClient(base_url="http://loss:8006", timeout=1.0, enabled=False, veto_p_loss_floor=0.9)
    result = await client.learn(feature_vector=[0.1], label="LOSS")
    assert result == {"ok": False, "skipped": True}
    await client.aclose()


@pytest.mark.asyncio
async def test_loss_classifier_client_learn_success():
    client = LossClassifierClient(base_url="http://loss:8006", timeout=1.0, enabled=True, veto_p_loss_floor=0.9)
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"ok": True, "n_train": 5})
    client._client.post = AsyncMock(return_value=response)
    result = await client.learn(feature_vector=[0.1], label="win", contract_id="c1", symbol="1HZ75V")
    assert result["ok"] is True
    payload = client._client.post.await_args.kwargs["json"]
    assert payload["label"] == "WIN"
    assert "schema_hash" in payload
    await client.aclose()


@pytest.mark.asyncio
async def test_loss_classifier_client_learn_non_dict_body():
    client = LossClassifierClient(base_url="http://loss:8006", timeout=1.0, enabled=True, veto_p_loss_floor=0.9)
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value="ok")
    client._client.post = AsyncMock(return_value=response)
    result = await client.learn(feature_vector=[0.1], label="LOSS")
    assert result == {"ok": True}
    await client.aclose()


@pytest.mark.asyncio
async def test_loss_classifier_client_learn_http_error_fallback():
    client = LossClassifierClient(base_url="http://loss:8006", timeout=1.0, enabled=True, veto_p_loss_floor=0.9)
    client._client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    result = await client.learn(feature_vector=[0.1], label="LOSS")
    assert result["ok"] is False
    assert "error" in result
    await client.aclose()


def test_loss_classifier_client_enabled_property():
    client = LossClassifierClient(base_url="http://loss:8006", timeout=1.0, enabled=True, veto_p_loss_floor=0.9)
    assert client.enabled is True
