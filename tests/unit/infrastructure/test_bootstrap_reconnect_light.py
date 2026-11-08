from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.deep_learning.dl_model_artifacts import (
    _light_infra_model_healthcheck,
    bootstrap_and_validate_models,
)


@pytest.mark.asyncio
async def test_light_infra_model_healthcheck_minio_ok():
    orch = MagicMock()
    orch.infra.enabled = True
    orch.model_store.head = AsyncMock(return_value=True)
    orch.config = {"infra": {}}
    await bootstrap_and_validate_models(orch, is_initial_boot=False)
    orch.model_store.head.assert_awaited_once()


@pytest.mark.asyncio
async def test_light_infra_model_healthcheck_minio_unavailable():
    orch = MagicMock()
    orch.infra.enabled = True
    orch.model_store.head = AsyncMock(return_value=False)
    orch.config = {}
    with pytest.raises(ConnectionError, match="MINIO"):
        await _light_infra_model_healthcheck(orch)


@pytest.mark.asyncio
async def test_bootstrap_validates_local_torchscript_then_light_reconnect(tmp_path):
    orch = MagicMock()
    orch.symbols = ["R_10"]
    orch.config = {
        "deep_learning": {"enabled": True, "use_torchscript": True, "arch": "tcn", "lookback": 48},
        "data_handler": {},
        "risk_management": {"params": {}},
        "infra": {},
    }
    orch.infra = MagicMock()
    orch.infra.enabled = False
    orch.model_store = MagicMock()
    orch.model_store.download_torchscript = AsyncMock(return_value=False)
    orch.model_store.load_manifest = AsyncMock(return_value={})
    orch._is_initial_boot = True
    ckpt = tmp_path / "R_10.pth"
    ckpt.write_bytes(b"x")
    ts_path = tmp_path / "R_10_ts.pt"
    ts_path.write_bytes(b"ts")

    with (
        patch(
            "src.application.services.deep_learning.dl_model_artifacts._light_infra_model_healthcheck",
            new_callable=AsyncMock,
        ) as light,
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.ensure_local_model_checkpoint",
            new_callable=AsyncMock,
            return_value=ckpt,
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts._scripted_path",
            return_value=ts_path,
        ),
    ):
        orch.model_store.sanity_check_torchscript = AsyncMock()
        await bootstrap_and_validate_models(orch, is_initial_boot=True)
        await bootstrap_and_validate_models(orch, is_initial_boot=False)
        await bootstrap_and_validate_models(orch, is_initial_boot=False)

    orch.model_store.sanity_check_torchscript.assert_awaited_once()
    assert light.await_count == 2
