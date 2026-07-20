from pathlib import Path

from scripts.operations.clean_workspace import build_safe_remove


def test_build_safe_remove_skips_docker_bind_mounts(tmp_path: Path):
    triton_model = tmp_path / "infra" / "docker" / "triton-models" / "R_10" / "1" / "model.pt"
    triton_model.parent.mkdir(parents=True)
    triton_model.write_bytes(b"pt")
    meta_model = tmp_path / "infra" / "docker" / "meta-models" / "meta_lgbm.pkl"
    meta_model.parent.mkdir(parents=True)
    meta_model.write_bytes(b"pkl")
    cache = tmp_path / "app" / ".pytest_cache"
    cache.mkdir(parents=True)

    safe_remove, preserved = build_safe_remove(tmp_path)
    assert len(preserved) == 2

    safe_remove(triton_model)
    safe_remove(meta_model)
    safe_remove(cache)

    assert triton_model.exists()
    assert meta_model.exists()
    assert not cache.exists()
