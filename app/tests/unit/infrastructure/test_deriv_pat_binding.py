import json
import urllib.error

import pytest

from src.infrastructure.api.deriv_pat_binding import (
    DerivPatBindingError,
    binding_path,
    discover_app_id_for_pat,
    load_binding,
    looks_like_pat,
    parse_deriv_pat,
    pat_fingerprint,
    probe_accounts_ok,
    read_candidate_app_ids,
    read_config_app_id,
    save_binding,
)


def test_parse_deriv_pat_composite():
    token, app = parse_deriv_pat("pat_abc123|app-77")
    assert token == "pat_abc123"
    assert app == "app-77"


def test_parse_deriv_pat_at_separator():
    token, app = parse_deriv_pat("pat_abc@app-88")
    assert token == "pat_abc"
    assert app == "app-88"


def test_parse_deriv_pat_plain():
    token, app = parse_deriv_pat("pat_abc123")
    assert token == "pat_abc123"
    assert app is None


def test_parse_deriv_pat_empty():
    token, app = parse_deriv_pat("  ")
    assert token == ""
    assert app is None


def test_looks_like_pat():
    assert looks_like_pat("pat_x")
    assert not looks_like_pat("token")


def test_save_and_load_binding(tmp_path, monkeypatch):
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)
    pat = "pat_test_token"
    save_binding(tmp_path, pat, "my-app")
    path = binding_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[pat_fingerprint(pat)]["app_id"] == "my-app"
    assert discover_app_id_for_pat(pat, tmp_path) == "my-app"
    assert load_binding(tmp_path, pat) == "my-app"


def test_load_binding_missing_and_invalid(tmp_path):
    assert load_binding(tmp_path, "pat_x") is None
    path = binding_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-json", encoding="utf-8")
    assert load_binding(tmp_path, "pat_x") is None
    path.write_text("[]", encoding="utf-8")
    assert load_binding(tmp_path, "pat_x") is None
    path.write_text('{"abc": "x"}', encoding="utf-8")
    assert load_binding(tmp_path, "pat_x") is None


def test_save_binding_merges_existing_dict(tmp_path):
    path = binding_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"other": {"app_id": "keep"}}', encoding="utf-8")
    save_binding(tmp_path, "pat_b", "app-2")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["other"]["app_id"] == "keep"
    assert load_binding(tmp_path, "pat_b") == "app-2"


def test_save_binding_merges_corrupt_file(tmp_path):
    path = binding_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{bad", encoding="utf-8")
    save_binding(tmp_path, "pat_a", "app-1")
    assert load_binding(tmp_path, "pat_a") == "app-1"


def test_discover_from_config_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "deriv_pat_app_id").write_text("cfg-app\n", encoding="utf-8")
    assert discover_app_id_for_pat("pat_x", tmp_path) == "cfg-app"


def test_read_config_app_id_txt_and_comment(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "deriv_pat_app_id.txt").write_text("real-id\n", encoding="utf-8")
    assert read_config_app_id(tmp_path) == "real-id"


def test_read_candidate_app_ids_filters_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_DERIV_APP_ID_CANDIDATES", "1089, cand-a, cand-a")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "deriv_pat_app_id.candidates").write_text(
        "cand-b\n# x\n\n16929\n",
        encoding="utf-8",
    )
    assert read_candidate_app_ids(tmp_path) == ["cand-a", "cand-b"]


def test_discover_explicit_and_env(monkeypatch, tmp_path):
    assert discover_app_id_for_pat("pat_x", tmp_path, explicit=" ex ") == "ex"
    monkeypatch.setenv("AETHER_DERIV_APP_ID", "env-app")
    assert discover_app_id_for_pat("pat_x", tmp_path) == "env-app"


def test_discover_from_pat_composite(tmp_path, monkeypatch):
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)
    assert discover_app_id_for_pat("pat_x|embedded", tmp_path) == "embedded"


def test_discover_probe_candidate(monkeypatch, tmp_path):
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "deriv_pat_app_id.candidates").write_text("probe-ok\n", encoding="utf-8")

    def ok(*_a, **_k):
        return True

    monkeypatch.setattr("src.infrastructure.api.deriv_pat_binding.probe_accounts_ok", ok)
    assert discover_app_id_for_pat("pat_probe", tmp_path) == "probe-ok"
    assert load_binding(tmp_path, "pat_probe") == "probe-ok"


def test_probe_accounts_ok_paths(monkeypatch):
    monkeypatch.setattr(
        "src.infrastructure.api.deriv_pat_binding.read_http_response",
        lambda *_a, **_k: b"ok",
    )
    assert probe_accounts_ok("pat", "app", "https://api.test", 1.0) is True

    def http_err(*_a, **_k):
        raise urllib.error.HTTPError("u", 401, "x", None, None)

    monkeypatch.setattr("src.infrastructure.api.deriv_pat_binding.read_http_response", http_err)
    assert probe_accounts_ok("pat", "app", "https://api.test", 1.0) is False

    def url_err(*_a, **_k):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("src.infrastructure.api.deriv_pat_binding.read_http_response", url_err)
    assert probe_accounts_ok("pat", "app", "https://api.test", 1.0) is False


def test_discover_missing_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AETHER_DERIV_APP_ID", raising=False)
    with pytest.raises(DerivPatBindingError):
        discover_app_id_for_pat("pat_missing", tmp_path)
