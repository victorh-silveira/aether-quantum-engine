"""Testes unitarios do entrypoint src/indicator/__main__.py."""

from __future__ import annotations

import pytest

from src.indicator import __main__ as indicator_main


def test_indicator_main_paths(tmp_path, monkeypatch):
    cfg = tmp_path / "config" / "settings.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('{"indicator": {"catalog_scope": "stock_indices"}}', encoding="utf-8")
    monkeypatch.setattr(indicator_main, "REPO_ROOT", tmp_path)
    called = {}

    async def fake_run(dsn, models_dir, ws_url, port, *, variant, scope):
        called["ok"] = (dsn, models_dir, ws_url, port, variant, scope)

    monkeypatch.setattr(indicator_main, "run_indicator", fake_run)
    indicator_main.main()
    assert called["ok"][5] == "stock_indices"

    cfg.write_text('{"indicator": {"catalog_scope": "invalid"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="catalog_scope"):
        indicator_main.main()
