import shutil
from pathlib import Path

from scripts.operations.clean_workspace import (
    clean_repo_data,
    clean_runtime_artifacts,
    is_docker_bind_mount,
    meta_models_root,
    triton_models_root,
)


def _tracking_remove():
    removed: list[Path] = []

    def safe_remove(path: Path) -> None:
        removed.append(path)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()

    return removed, safe_remove


def test_triton_models_root_points_to_docker_bind_mount(tmp_path: Path):
    root = triton_models_root(tmp_path)
    assert root == tmp_path / "infra" / "docker" / "triton-models"


def test_meta_models_root_points_to_docker_bind_mount(tmp_path: Path):
    root = meta_models_root(tmp_path)
    assert root == tmp_path / "infra" / "docker" / "meta-models"


def test_is_docker_bind_mount_detects_nested_artifacts(tmp_path: Path):
    triton_model = tmp_path / "infra" / "docker" / "triton-models" / "RDBULL" / "1" / "model.pt"
    meta_model = tmp_path / "infra" / "docker" / "meta-models" / "meta_lgbm.pkl"
    other = tmp_path / "infra" / "docker" / "redis.conf"
    assert is_docker_bind_mount(triton_model, tmp_path)
    assert is_docker_bind_mount(meta_model, tmp_path)
    assert not is_docker_bind_mount(other, tmp_path)


def test_clean_runtime_artifacts_preserves_docker_bind_mounts(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    session = data_root / "session_state.json"
    session.write_text("{}", encoding="utf-8")
    triton_model = tmp_path / "infra" / "docker" / "triton-models" / "RDBULL" / "1" / "model.pt"
    triton_model.parent.mkdir(parents=True)
    triton_model.write_bytes(b"pt")
    triton_config = triton_model.parent.parent / "config.pbtxt"
    triton_config.write_text("name", encoding="utf-8")
    meta_model = tmp_path / "infra" / "docker" / "meta-models" / "meta_lgbm.pkl"
    meta_model.parent.mkdir(parents=True)
    meta_model.write_bytes(b"pkl")

    removed, safe_remove = _tracking_remove()
    clean_runtime_artifacts(tmp_path, safe_remove)

    assert session in removed
    assert triton_model not in removed
    assert meta_model not in removed
    assert not session.exists()
    assert triton_model.exists()
    assert triton_config.exists()
    assert meta_model.exists()
    assert triton_model.parent.is_dir()


def test_clean_repo_data_preserves_deriv_bindings(tmp_path: Path):
    data_root = tmp_path / "data"
    dl_root = data_root / "dl"
    dl_root.mkdir(parents=True)
    checkpoint = dl_root / "RDBULL.pth"
    checkpoint.write_bytes(b"pth")
    state = data_root / "state.json"
    state.write_text("{}", encoding="utf-8")
    deriv = data_root / "deriv"
    deriv.mkdir()
    binding = deriv / "pat_bindings.json"
    binding.write_text("{}", encoding="utf-8")

    removed, safe_remove = _tracking_remove()
    clean_repo_data(data_root, safe_remove)

    assert dl_root in removed
    assert state in removed
    assert not state.exists()
    assert binding.exists()
    assert deriv.exists()


def test_clean_repo_data_noop_when_data_missing(tmp_path: Path):
    removed, safe_remove = _tracking_remove()
    clean_repo_data(tmp_path / "data", safe_remove)
    assert removed == []
