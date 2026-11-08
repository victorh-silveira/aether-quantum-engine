from src.application.services.deep_learning.dl_model_checkpoint import load_model_checkpoint


def test_safe_load_non_existent_checkpoint(tmp_path):
    res = load_model_checkpoint(tmp_path / "does_not_exist.pt")
    assert res is None
