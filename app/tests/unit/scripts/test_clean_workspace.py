from pathlib import Path

from scripts.operations.clean_workspace import _repo_root_for_bash, build_safe_remove


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
