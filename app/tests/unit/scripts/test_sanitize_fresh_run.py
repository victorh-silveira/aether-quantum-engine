from pathlib import Path

from scripts.operations.sanitize_fresh_run import sanitize_fresh_run


def test_sanitize_fresh_run_clears_checkpoints_and_keeps_deriv(tmp_path: Path):
    dl = tmp_path / "data" / "dl"
    dl.mkdir(parents=True)
    (dl / "R_10.pth").write_bytes(b"pth")
    (dl / "R_10_ts.pt").write_bytes(b"ts")
    state = tmp_path / "data" / "session_state.json"
    state.write_text("{}", encoding="utf-8")
    deriv = tmp_path / "data" / "deriv"
    deriv.mkdir()
    binding = deriv / "pat_bindings.json"
    binding.write_text("{}", encoding="utf-8")
    meta = tmp_path / "infra" / "docker" / "meta-models"
    meta.mkdir(parents=True)
    (meta / "meta_lgbm.pkl").write_bytes(b"meta")
    (meta / "meta_learn_buffer.pkl").write_bytes(b"buf")
    loss = tmp_path / "infra" / "docker" / "loss-models"
    loss.mkdir(parents=True)
    (loss / "learn_buffer.pkl").write_bytes(b"loss")
    triton = tmp_path / "infra" / "docker" / "triton-models" / "R_10" / "1"
    triton.mkdir(parents=True)
    (triton / "model.pt").write_bytes(b"pt")
    (triton / ".gitkeep").write_text("", encoding="utf-8")

    counts = sanitize_fresh_run(tmp_path)

    assert counts["dl"] == 2
    assert counts["data_runtime"] == 1
    assert counts["meta"] == 2
    assert counts["loss"] == 1
    assert counts["triton"] == 1
    assert not (dl / "R_10.pth").exists()
    assert not state.exists()
    assert binding.exists()
    assert not (meta / "meta_lgbm.pkl").exists()
    assert not (loss / "learn_buffer.pkl").exists()
    assert not (triton / "model.pt").exists()
    assert (triton / ".gitkeep").exists()


def test_sanitize_fresh_run_keep_meta_bundle(tmp_path: Path):
    meta = tmp_path / "infra" / "docker" / "meta-models"
    meta.mkdir(parents=True)
    bundle = meta / "meta_lgbm.pkl"
    bundle.write_bytes(b"meta")
    buf = meta / "meta_learn_buffer.pkl"
    buf.write_bytes(b"buf")
    (tmp_path / "data" / "dl").mkdir(parents=True)
    (tmp_path / "data" / "dl" / "R_10.pth").write_bytes(b"pth")

    counts = sanitize_fresh_run(tmp_path, keep_meta_bundle=True)

    assert counts["dl"] == 1
    assert counts["meta"] == 1
    assert bundle.exists()
    assert not buf.exists()
