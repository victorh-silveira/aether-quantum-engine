"""Testes unitarios para src/shared/*, src/indicator/__main__.py e train_cli.py."""

from __future__ import annotations

import asyncio
import importlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator import train_cli
from src.shared import paths, settings_io


def test_shared_paths_and_settings_io(tmp_path, monkeypatch):
    filtered = [p for p in sys.path if p not in (str(paths.REPO_ROOT), str(paths.SRC_ROOT))]
    monkeypatch.setattr(sys, "path", filtered)
    importlib.reload(paths)
    assert str(paths.SRC_ROOT) in sys.path
    assert paths.repo_path("config", "settings.json").name == "settings.json"
    cfg = settings_io.load_settings_json()
    assert isinstance(cfg, dict)

    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(settings_io, "repo_path", lambda *args: bad)
    with pytest.raises(ValueError, match="settings.json invalido"):
        settings_io.load_settings_json()


@pytest.mark.asyncio
async def test_train_cli_main_paths(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(train_cli, "load_engine_config", lambda **kw: ({}, logger))
    monkeypatch.setattr(train_cli, "create_authenticated_auth", lambda c, log: None)
    assert await train_cli.main() == 1

    monkeypatch.setattr(train_cli, "create_authenticated_auth", lambda c, log: MagicMock())
    orch = MagicMock()
    orch.symbols = ["OTC_SSMI"]
    orch.stop = AsyncMock()
    monkeypatch.setattr(train_cli, "Orchestrator", lambda c, a: orch)

    orch.run_training = AsyncMock(return_value=True)
    assert await train_cli.main() == 0

    orch.run_training = AsyncMock(return_value=False)
    assert await train_cli.main() == 1

    orch.run_training = AsyncMock(side_effect=asyncio.CancelledError())
    assert await train_cli.main() == 130
