"""Testes do universo de simbolos do sweep."""

from __future__ import annotations

from pathlib import Path

from src.application.services.deep_learning.tf_sweep_symbols import (
    DEFAULT_SWEEP_SYMBOLS,
    clear_other_live_checkpoints,
    patch_settings_for_symbol,
    resolve_sweep_symbols,
    write_trading_symbols_module,
)


def test_resolve_sweep_symbols_default_and_filter():
    assert DEFAULT_SWEEP_SYMBOLS == ("1HZ75V",)
    assert resolve_sweep_symbols({}) == ["1HZ75V"]
    assert resolve_sweep_symbols({"symbols": ["r_25", "R_25", "R_50"]}) == ["R_25", "R_50"]


def test_patch_and_write_drift(tmp_path: Path):
    settings = {"deep_learning": {}}
    patch_settings_for_symbol(settings, "R_75")
    assert settings["anchor"] == "R_75"
    assert settings["symbols"] == ["R_75"]
    assert settings["deep_learning"]["train_symbols"] == ["R_75"]
    path = write_trading_symbols_module("R_75", path=tmp_path / "drift_symbols.py")
    text = path.read_text(encoding="utf-8")
    assert 'TRADING_SYMBOLS: tuple[str, ...] = ("R_75",)' in text
    assert 'DEFAULT_ANCHOR = "R_75"' in text


def test_clear_other_live_checkpoints(tmp_path: Path):
    (tmp_path / "R_10.pth").write_bytes(b"a")
    (tmp_path / "R_25.pth").write_bytes(b"b")
    (tmp_path / "R_25_ts.pt").write_bytes(b"c")
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
    removed = clear_other_live_checkpoints(tmp_path, "R_25")
    assert (tmp_path / "R_25.pth").is_file()
    assert (tmp_path / "R_25_ts.pt").is_file()
    assert (tmp_path / "notes.txt").is_file()
    assert not (tmp_path / "R_10.pth").exists()
    assert any(p.name == "R_10.pth" for p in removed)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "other.pth").write_bytes(b"x")
    still = clear_other_live_checkpoints(tmp_path, "R_25")
    assert still == []
    assert nested.is_dir()
    missing = tmp_path / "nao_existe"
    assert clear_other_live_checkpoints(missing, "R_25") == []
