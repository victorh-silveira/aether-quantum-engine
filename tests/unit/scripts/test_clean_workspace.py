from pathlib import Path

from scripts.operations import clean_workspace
from scripts.operations.clean_workspace import _repo_root_for_bash, build_safe_remove


def test_test_profile_limits_parallel_workers_on_windows(monkeypatch):
    monkeypatch.setattr(clean_workspace.sys, "platform", "win32")
    monkeypatch.setattr(clean_workspace, "_available_ram_gb", lambda: 64.0)
    assert clean_workspace.resolve_test_execution_profile().parallel_workers == 0


def test_test_profile_limits_parallel_workers_on_linux(monkeypatch):
    monkeypatch.setattr(clean_workspace.sys, "platform", "linux")
    monkeypatch.setattr(clean_workspace, "_available_ram_gb", lambda: 64.0)
    assert clean_workspace.resolve_test_execution_profile().parallel_workers <= 2


def test_stage_test_uses_bounded_streamed_process(monkeypatch):
    calls = []
    monkeypatch.setattr(
        clean_workspace,
        "resolve_test_execution_profile",
        lambda: clean_workspace.TestExecutionProfile("seguro", 0, 8.0),
    )
    monkeypatch.setattr(clean_workspace, "_purge_coverage_artifacts", lambda _root: None)
    monkeypatch.setattr(clean_workspace, "_release_parent_memory", lambda: None)
    monkeypatch.setattr(clean_workspace.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    clean_workspace.stage_test()
    assert calls[0][0][0][calls[0][0][0].index("-n") + 1] == "0"
    assert calls[0][1]["timeout"] == 600
    assert "capture_output" not in calls[0][1]


def test_gitleaks_source_uses_linux_path_for_wsl_checkout():
    assert _repo_root_for_bash(Path("//wsl.localhost/kali-linux/home/victor/aether")) == "/home/victor/aether"


def test_build_safe_remove_skips_docker_bind_mounts(tmp_path: Path):
    meta_model = tmp_path / "infra" / "docker" / "meta-models" / "meta_lgbm.pkl"
    meta_model.parent.mkdir(parents=True)
    meta_model.write_bytes(b"pkl")
    cache = tmp_path / "app" / ".pytest_cache"
    cache.mkdir(parents=True)

    safe_remove, preserved = build_safe_remove(tmp_path)
    assert len(preserved) == 1

    safe_remove(meta_model)
    safe_remove(cache)

    assert meta_model.exists()
    assert not cache.exists()
