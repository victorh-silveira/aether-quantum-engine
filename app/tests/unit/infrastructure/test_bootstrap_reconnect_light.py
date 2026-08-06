from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.deep_learning.dl_model_artifacts import (
    _bootstrap_triton_models,
    _light_infra_model_healthcheck,
    bootstrap_and_validate_models,
)


@pytest.mark.asyncio
async def test_light_infra_model_healthcheck_minio_and_triton():
    orch = MagicMock()
    orch.infra.enabled = True
    orch.model_store.head = AsyncMock(return_value=True)
    orch.config = {"infra": {"triton": {"enabled": True}}}
    with patch(
        "src.application.services.deep_learning.dl_model_artifacts.verify_triton_healthcheck_async",
        new_callable=AsyncMock,
    ) as health:
        await bootstrap_and_validate_models(orch, is_initial_boot=False)
    orch.model_store.head.assert_awaited_once()
    health.assert_awaited_once_with(orch.config)


@pytest.mark.asyncio
async def test_light_infra_model_healthcheck_minio_unavailable():
    orch = MagicMock()
    orch.infra.enabled = True
    orch.model_store.head = AsyncMock(return_value=False)
    orch.config = {}
    with pytest.raises(ConnectionError, match="MINIO"):
        await _light_infra_model_healthcheck(orch)


@pytest.mark.asyncio
async def test_stress_inference_called_once_across_boot_and_reconnect(tmp_path):
    orch = MagicMock()
    orch.symbols = ["OTC_SPC"]
    orch.config = {
        "deep_learning": {"enabled": True, "use_torchscript": True, "arch": "tcn", "lookback": 48},
        "data_handler": {},
        "risk_management": {"params": {}},
        "infra": {"triton": {"enabled": True}},
    }
    orch.infra = MagicMock()
    orch.infra.enabled = False
    orch.model_store = MagicMock()
    orch.model_store.download_torchscript = AsyncMock(return_value=False)
    orch.model_store.load_manifest = AsyncMock(return_value={})
    orch._is_initial_boot = True
    ckpt = tmp_path / "OTC_SPC.pth"
    ckpt.write_bytes(b"x")
    ts_path = tmp_path / "OTC_SPC_ts.pt"
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
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.sync_all_symbols_to_triton",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.verify_triton_schema_alignment_async",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.verify_triton_stressed_inference_async",
            new_callable=AsyncMock,
        ) as stress,
    ):
        orch.model_store.sanity_check_torchscript = AsyncMock()
        await bootstrap_and_validate_models(orch, is_initial_boot=True)
        await bootstrap_and_validate_models(orch, is_initial_boot=False)
        await bootstrap_and_validate_models(orch, is_initial_boot=False)

    assert stress.await_count == 1
    assert light.await_count == 2


@pytest.mark.asyncio
async def test_bootstrap_train_mode_skips_triton_verify_without_torchscript(tmp_path):
    orch = MagicMock()
    orch.symbols = ["OTC_SPC", "R_50"]
    orch.config = {
        "orchestrator": {"engine_mode": "train"},
        "deep_learning": {"enabled": True, "use_torchscript": True, "arch": "tcn", "lookback": 48},
        "data_handler": {},
        "risk_management": {"params": {}},
        "infra": {"triton": {"enabled": True}},
    }
    orch.infra = MagicMock()
    orch.infra.enabled = True
    orch.model_store = MagicMock()
    orch.model_store.download_torchscript = AsyncMock(return_value=False)
    orch.model_store.load_manifest = AsyncMock(return_value={})
    ckpt = tmp_path / "OTC_SPC.pth"
    ckpt.write_bytes(b"x")
    with (
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.ensure_local_model_checkpoint",
            new_callable=AsyncMock,
            return_value=ckpt,
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts._scripted_path",
            return_value=tmp_path / "OTC_SPC_ts.pt",
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.sync_all_symbols_to_triton",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.verify_triton_schema_alignment_async",
            new_callable=AsyncMock,
        ) as schema,
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.verify_triton_stressed_inference_async",
            new_callable=AsyncMock,
        ) as stress,
    ):
        await bootstrap_and_validate_models(orch, is_initial_boot=True)
    schema.assert_not_awaited()
    stress.assert_not_awaited()


@pytest.mark.asyncio
async def test_bootstrap_execute_mode_requires_torchscript_for_triton(tmp_path):
    orch = MagicMock()
    orch.symbols = ["OTC_SPC"]
    orch.config = {
        "deep_learning": {"enabled": True, "use_torchscript": True, "arch": "tcn", "lookback": 48},
        "data_handler": {},
        "risk_management": {"params": {}},
        "infra": {"triton": {"enabled": True}},
    }
    orch.infra = MagicMock()
    orch.infra.enabled = False
    orch.model_store = object()
    ckpt = tmp_path / "OTC_SPC.pth"
    ckpt.write_bytes(b"x")
    with (
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.ensure_local_model_checkpoint",
            new_callable=AsyncMock,
            return_value=ckpt,
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts._scripted_path",
            return_value=tmp_path / "OTC_SPC_ts.pt",
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.sync_all_symbols_to_triton",
            new_callable=AsyncMock,
        ),
        pytest.raises(ConnectionError, match="TorchScript ausente"),
    ):
        await bootstrap_and_validate_models(orch, is_initial_boot=True)


@pytest.mark.asyncio
async def test_bootstrap_marks_triton_ready_from_local_torchscript_without_sanity(tmp_path):
    orch = MagicMock()
    orch.symbols = ["OTC_SPC"]
    orch.config = {
        "deep_learning": {"enabled": True, "use_torchscript": False, "arch": "tcn", "lookback": 48},
        "data_handler": {},
        "risk_management": {"params": {}},
        "infra": {"triton": {"enabled": True}},
    }
    orch.infra = MagicMock()
    orch.infra.enabled = False
    orch.model_store = object()
    ckpt = tmp_path / "OTC_SPC.pth"
    ckpt.write_bytes(b"x")
    ts_path = tmp_path / "OTC_SPC_ts.pt"
    ts_path.write_bytes(b"ts")
    with (
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.ensure_local_model_checkpoint",
            new_callable=AsyncMock,
            return_value=ckpt,
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts._scripted_path",
            return_value=ts_path,
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.sync_all_symbols_to_triton",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.verify_triton_schema_alignment_async",
            new_callable=AsyncMock,
        ) as schema,
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.verify_triton_stressed_inference_async",
            new_callable=AsyncMock,
        ) as stress,
    ):
        await bootstrap_and_validate_models(orch, is_initial_boot=True)
    schema.assert_awaited_once()
    stress.assert_awaited_once()


@pytest.mark.asyncio
async def test_bootstrap_triton_skips_probe_when_symbol_list_empty():
    orch = MagicMock()
    orch.symbols = []
    orch.config = {"infra": {"triton": {"enabled": True}}}
    with (
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.sync_all_symbols_to_triton",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.verify_triton_schema_alignment_async",
            new_callable=AsyncMock,
        ) as schema,
    ):
        await _bootstrap_triton_models(orch, lookback=48, triton_ready=True)
    schema.assert_not_awaited()
