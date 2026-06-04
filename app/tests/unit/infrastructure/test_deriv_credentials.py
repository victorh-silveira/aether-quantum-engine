import json

from src.infrastructure.api.deriv_credentials import (
    _load_settings_deriv_app_id,
    is_legacy_deriv_app_id,
    looks_like_pat,
    persist_deriv_app_id,
    resolve_deriv_app_id,
)
from src.infrastructure.api.deriv_pat_binding import DerivPatBindingError


def test_resolve_deriv_app_id_from_env(monkeypatch):
    monkeypatch.setenv("AETHER_DERIV_APP_ID", "app-99")
    assert resolve_deriv_app_id() == "app-99"


def test_resolve_deriv_app_id_explicit_over_env(monkeypatch):
    monkeypatch.setenv("AETHER_DERIV_APP_ID", "from-env")
    assert resolve_deriv_app_id("from-arg") == "from-arg"


def test_resolve_deriv_app_id_from_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)
    monkeypatch.delenv("DERIV_APP_ID", raising=False)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "settings.json").write_text(
        json.dumps({"api_config": {"deriv_app_id": "from-json"}}),
        encoding="utf-8",
    )
    assert resolve_deriv_app_id(repo_root=tmp_path) == "from-json"


def test_legacy_and_pat_helpers():
    assert is_legacy_deriv_app_id("1089")
    assert not is_legacy_deriv_app_id("new-id")
    assert looks_like_pat("pat_abc")
    assert not looks_like_pat("12345")


def test_persist_deriv_app_id(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AETHER_DERIV_PAT=pat_x\n", encoding="utf-8")
    persist_deriv_app_id(tmp_path, "saved-id")
    text = env.read_text(encoding="utf-8")
    assert "AETHER_DERIV_APP_ID=saved-id" in text
    assert "AETHER_DERIV_PAT=pat_x" in text


def test_persist_deriv_app_id_replaces_existing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AETHER_DERIV_APP_ID=old\n", encoding="utf-8")
    persist_deriv_app_id(tmp_path, "new-id")
    assert env.read_text(encoding="utf-8").count("AETHER_DERIV_APP_ID=") == 1
    assert "new-id" in env.read_text(encoding="utf-8")


def test_load_settings_deriv_app_id_edge_cases(tmp_path):
    assert _load_settings_deriv_app_id(None) == ""
    assert _load_settings_deriv_app_id(tmp_path) == ""
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "settings.json").write_text("[]", encoding="utf-8")
    assert _load_settings_deriv_app_id(tmp_path) == ""
    (cfg / "settings.json").write_text('{"api_config": []}', encoding="utf-8")
    assert _load_settings_deriv_app_id(tmp_path) == ""
    (cfg / "settings.json").write_text("not-json", encoding="utf-8")
    assert _load_settings_deriv_app_id(tmp_path) == ""


def test_resolve_deriv_app_id_from_pat(tmp_path, monkeypatch):
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "deriv_pat_app_id").write_text("from-pat-cfg\n", encoding="utf-8")
    assert resolve_deriv_app_id(pat="pat_x", repo_root=tmp_path) == "from-pat-cfg"


def test_resolve_deriv_app_id_pat_binding_error_falls_back(monkeypatch, tmp_path):
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)

    def boom(*_a, **_k):
        raise DerivPatBindingError("x")

    monkeypatch.setattr("src.infrastructure.api.deriv_credentials.discover_app_id_for_pat", boom)
    assert (
        resolve_deriv_app_id(pat="pat_x", repo_root=tmp_path, config={"api_config": {"deriv_app_id": "cfg"}}) == "cfg"
    )


def test_resolve_deriv_app_id_from_config_dict(monkeypatch):
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)
    assert resolve_deriv_app_id(config={"api_config": {"deriv_app_id": " inline "}}) == "inline"


def test_resolve_deriv_app_id_deriv_env(monkeypatch):
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)
    monkeypatch.setenv("DERIV_APP_ID", "deriv-env")
    assert resolve_deriv_app_id() == "deriv-env"
