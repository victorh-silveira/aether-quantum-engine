import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "src"


def _py_files():
    return list(ROOT.rglob("*.py"))


def test_no_active_direction_invert_api_exports():
    forbidden = {
        "apply_configured_direction_invert",
        "should_flip_direction",
        "flipped_direction",
        "apply_meta_direction_flip",
        "apply_recovery_direction_flip",
        "inject_recovery_hedge_candidates",
        "recovery_hedge_target",
        "build_forced_direction_candidate",
        "build_forced_recovery_candidate",
    }
    found = set()
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in forbidden:
                found.add(node.name)
    assert found == set()


def test_settings_has_no_invert_flags():
    settings = (Path(__file__).resolve().parents[4] / "config" / "settings.json").read_text(encoding="utf-8")
    assert "invert_execution_direction" not in settings
    assert "recovery_flip_direction_after_loss" not in settings
    assert "side_equilibrium" in settings
