"""Testes de store/bind/pop do vetor meta, learn sync e feed de settlement."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.meta_classifier_cross_symbol import META_FEATURE_DIM
from src.application.services.meta_classifier_vectors import (
    bind_meta_feature_vector_to_contract,
    pop_meta_feature_vector,
    store_meta_feature_vector,
)
from src.application.services.orchestrator.settlement_outcome import _feed_meta_classifier_learn
from src.infrastructure.inference.meta_classifier_client import MetaClassifierClient
from src.infrastructure.inference.meta_classifier_pool import learn_meta_via_config_sync


def test_meta_vector_store_bind_pop():
    orch = SimpleNamespace()
    store_meta_feature_vector(orch, "R_10", [0.1] * 43)
    bind_meta_feature_vector_to_contract(orch, "R_10", 42)
    popped = pop_meta_feature_vector(orch, "R_10", 42)
    assert popped is not None
    assert len(popped) == 43
    assert pop_meta_feature_vector(orch, "R_10", 42) is None


def test_meta_vector_early_returns():
    orch = SimpleNamespace()
    store_meta_feature_vector(orch, "", [0.1] * 43)
    store_meta_feature_vector(orch, "R_10", [])
    store_meta_feature_vector(orch, "R_10", None)
    assert getattr(orch, "_meta_clf_vectors", None) in (None, {})
    bind_meta_feature_vector_to_contract(orch, "R_10", 1)
    assert pop_meta_feature_vector(orch, "R_10", 1) is None
    store_meta_feature_vector(orch, "R_10", [0.2] * 43)
    assert pop_meta_feature_vector(orch, "R_10", 99) == [0.2] * 43


def test_should_retrain_meta_requires_lgbm_min_samples():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[4] / "infra" / "docker" / "meta-classifier" / "learn_runtime.py"
    spec = importlib.util.spec_from_file_location("meta_learn_runtime", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.meta_retrain_floor(1) == 2
    assert mod.should_retrain_meta(buffer_n=1, retrain_min_n=1) is False
    assert mod.should_retrain_meta(buffer_n=2, retrain_min_n=1) is True
    assert mod.should_retrain_meta(buffer_n=1, retrain_min_n=2) is False
    assert mod.should_retrain_meta(buffer_n=2, retrain_min_n=2) is True
    assert mod.should_retrain_meta(buffer_n=0, retrain_min_n=2) is False


def test_feed_meta_classifier_learn_paths(caplog):
    import logging

    orch = SimpleNamespace(config={"infra": {"meta_classifier": {"enabled": True}}})
    _feed_meta_classifier_learn(orch, "R_10", profit=1.0, stake=10.0, contract_id=1)
    store_meta_feature_vector(orch, "R_10", [0.1] * META_FEATURE_DIM)
    bind_meta_feature_vector_to_contract(orch, "R_10", 2)
    orch.config = None
    _feed_meta_classifier_learn(orch, "R_10", profit=1.0, stake=10.0, contract_id=2)
    orch.config = {"infra": {"meta_classifier": {"enabled": True}}}
    store_meta_feature_vector(orch, "R_10", [0.1] * META_FEATURE_DIM)
    bind_meta_feature_vector_to_contract(orch, "R_10", 3)
    with patch(
        "src.application.services.orchestrator.settlement_outcome.learn_meta_via_config_sync",
        return_value={"skipped": True},
    ):
        with caplog.at_level(logging.WARNING):
            _feed_meta_classifier_learn(orch, "R_10", profit=-5.0, stake=10.0, contract_id=3)
        assert "META || LEARN falhou" in caplog.text
    store_meta_feature_vector(orch, "R_10", [0.1] * META_FEATURE_DIM)
    bind_meta_feature_vector_to_contract(orch, "R_10", 4)
    with patch(
        "src.application.services.orchestrator.settlement_outcome.learn_meta_via_config_sync",
        return_value={"ok": True, "buffer_n": 1, "retrained": True, "retrain_detail": "ok"},
    ):
        with caplog.at_level(logging.INFO):
            _feed_meta_classifier_learn(orch, "R_10", profit=7.2, stake=10.0, contract_id=4)
        assert "META || LEARN" in caplog.text
        assert orch._last_meta_clf_learn.startswith("target=")
    store_meta_feature_vector(orch, "R_10", [0.1] * META_FEATURE_DIM)
    bind_meta_feature_vector_to_contract(orch, "R_10", 5)
    with patch(
        "src.application.services.orchestrator.settlement_outcome.learn_meta_via_config_sync",
        return_value=None,
    ):
        _feed_meta_classifier_learn(orch, "R_10", profit=1.0, stake=10.0, contract_id=5)


def test_learn_meta_via_config_sync_branches():
    assert learn_meta_via_config_sync(
        {"infra": {"meta_classifier": {"enabled": False}}}, feature_vector=[0.0] * 43, target=0.1
    )["skipped"]
    with patch(
        "src.infrastructure.inference.meta_classifier_pool.get_meta_classifier_client",
        new_callable=AsyncMock,
    ) as mock_get:
        client = MagicMock()
        client.learn = AsyncMock(return_value={"ok": True, "retrained": True})
        mock_get.return_value = client
        out = learn_meta_via_config_sync(
            {"infra": {"meta_classifier": {"enabled": True, "online_learn": True, "timeout_seconds": 1.0}}},
            feature_vector=[0.0] * 43,
            target=0.5,
            contract_id="9",
            symbol="R_10",
        )
        assert out["retrained"] is True
    skipped = learn_meta_via_config_sync(
        {"infra": {"meta_classifier": {"enabled": True, "online_learn": False}}},
        feature_vector=[0.0] * 43,
        target=0.1,
    )
    assert skipped.get("skipped") is True


@pytest.mark.asyncio
async def test_meta_client_learn_paths():
    import httpx

    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=False)
    assert (await client.learn(feature_vector=[0.0] * 43, target=0.1))["skipped"] is True
    client = MetaClassifierClient(base_url="http://meta:8005", timeout=1.0, enabled=True)
    client._client.post = AsyncMock(
        return_value=MagicMock(raise_for_status=MagicMock(), json=MagicMock(return_value={"ok": True, "retrained": 1}))
    )
    out = await client.learn(feature_vector=[0.0] * META_FEATURE_DIM, target=0.72, contract_id="1", symbol="R_10")
    assert out["ok"] is True
    client._client.post = AsyncMock(side_effect=httpx.TimeoutException("x"))
    err = await client.learn(feature_vector=[0.0] * META_FEATURE_DIM, target=-1.0)
    assert "error" in err
    client._client.post = AsyncMock(
        return_value=MagicMock(raise_for_status=MagicMock(), json=MagicMock(return_value=["bad"]))
    )
    bad = await client.learn(feature_vector=[0.0] * META_FEATURE_DIM, target=0.0)
    assert bad.get("error") == "invalid_json"
