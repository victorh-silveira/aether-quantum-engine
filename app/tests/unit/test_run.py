import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

import run
from src.application.services.orchestrator.engine_session import (
    create_authenticated_auth,
    load_engine_config,
)


_MIN_CONFIG = {
    "logging": {"log_file": "logs/test.log"},
    "trading": {"mode": "demo"},
    "orchestrator": {},
}


def test_emit_fatal_startup_error_skips_closed_stderr():
    stderr = MagicMock()
    stderr.closed = True
    with patch.object(sys, "stderr", stderr):
        run._emit_fatal_startup_error(RuntimeError("boom"))


def test_emit_fatal_startup_error_writes_to_stderr():
    stderr = MagicMock()
    stderr.closed = False
    with (
        patch.object(sys, "stderr", stderr),
        patch("builtins.print") as print_mock,
    ):
        run._emit_fatal_startup_error(RuntimeError("boom"))
    print_mock.assert_called_once()


def test_emit_fatal_startup_error_swallows_print_failure():
    with patch("builtins.print", side_effect=OSError("stream down")):
        run._emit_fatal_startup_error(RuntimeError("boom"))


def test_emit_fatal_startup_error_skips_missing_stderr():
    with patch.object(sys, "stderr", None):
        run._emit_fatal_startup_error(RuntimeError("boom"))


@pytest.mark.asyncio
async def test_main_returns_130_on_keyboard_interrupt():
    orch = MagicMock()
    orch.run = AsyncMock(side_effect=KeyboardInterrupt())
    orch.close_infrastructure_connections = AsyncMock()
    with (
        patch("run.load_engine_config", return_value=({}, MagicMock())),
        patch("run.create_authenticated_auth", return_value=MagicMock()),
        patch("run.Orchestrator", return_value=orch),
    ):
        assert await run.main() == 130
    orch.close_infrastructure_connections.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_returns_130_on_cancelled_error():
    orch = MagicMock()
    orch.run = AsyncMock(side_effect=asyncio.CancelledError())
    orch.close_infrastructure_connections = AsyncMock()
    with (
        patch("run.load_engine_config", return_value=({}, MagicMock())),
        patch("run.create_authenticated_auth", return_value=MagicMock()),
        patch("run.Orchestrator", return_value=orch),
    ):
        assert await run.main() == 130


@pytest.mark.asyncio
async def test_main_returns_1_when_auth_missing():
    with (
        patch("run.load_engine_config", return_value=({}, MagicMock())),
        patch("run.create_authenticated_auth", return_value=None),
    ):
        assert await run.main() == 1


@pytest.mark.asyncio
async def test_main_returns_0_on_stop_win():
    orch = MagicMock()
    orch.run = AsyncMock()
    orch.close_infrastructure_connections = AsyncMock()
    orch.shutdown_reason = "stop_win"
    orch.running = True
    orch.risk_manager.total_session_profit = 12.5
    with (
        patch("run.load_engine_config", return_value=({}, MagicMock())),
        patch("run.create_authenticated_auth", return_value=MagicMock()),
        patch("run.Orchestrator", return_value=orch),
    ):
        assert await run.main() == 0


@pytest.mark.asyncio
async def test_main_returns_1_when_loop_ends_before_running():
    orch = MagicMock()
    orch.run = AsyncMock()
    orch.close_infrastructure_connections = AsyncMock()
    orch.shutdown_reason = None
    orch.running = False
    with (
        patch("run.load_engine_config", return_value=({}, MagicMock())),
        patch("run.create_authenticated_auth", return_value=MagicMock()),
        patch("run.Orchestrator", return_value=orch),
    ):
        assert await run.main() == 1


def test_load_engine_config_reads_settings():
    payload = json.dumps(_MIN_CONFIG)
    with (
        patch("src.application.services.orchestrator.engine_session.os.chdir"),
        patch(
            "src.application.services.orchestrator.engine_session.repo_path",
            return_value=MagicMock(open=mock_open(read_data=payload)),
        ),
        patch("src.application.services.orchestrator.engine_session.setup_logger", return_value=MagicMock()),
    ):
        config, logger = load_engine_config()
    assert config["trading"]["mode"] == "demo"
    assert logger is not None


def test_load_engine_config_logs_risk_validation_issues():
    payload = json.dumps(_MIN_CONFIG)
    logger = MagicMock()
    with (
        patch("src.application.services.orchestrator.engine_session.os.chdir"),
        patch(
            "src.application.services.orchestrator.engine_session.repo_path",
            return_value=MagicMock(open=mock_open(read_data=payload)),
        ),
        patch("src.application.services.orchestrator.engine_session.setup_logger", return_value=logger),
        patch(
            "src.application.services.orchestrator.engine_session.validate_engine_risk_config",
            return_value=["kelly.max_stake_pct fora de (0, 0.10]: 0.5"],
        ),
    ):
        load_engine_config()
    logger.warning.assert_called_once_with("CFG_RISK || %s", "kelly.max_stake_pct fora de (0, 0.10]: 0.5")


def test_create_authenticated_auth_returns_none_without_pat():
    auth = MagicMock()
    auth.get_pat.return_value = ""
    with patch("src.application.services.orchestrator.engine_session.AuthManager", return_value=auth):
        assert create_authenticated_auth(_MIN_CONFIG, MagicMock()) is None


def test_create_authenticated_auth_returns_none_when_rest_client_fails():
    auth = MagicMock()
    auth.get_pat.return_value = "pat_token"
    auth.rest_client.side_effect = RuntimeError("rest down")
    with patch("src.application.services.orchestrator.engine_session.AuthManager", return_value=auth):
        assert create_authenticated_auth(_MIN_CONFIG, MagicMock()) is None


def test_create_authenticated_auth_returns_manager_when_pat_valid():
    auth = MagicMock()
    auth.get_pat.return_value = "pat_token"
    with patch("src.application.services.orchestrator.engine_session.AuthManager", return_value=auth):
        assert create_authenticated_auth(_MIN_CONFIG, MagicMock()) is auth
