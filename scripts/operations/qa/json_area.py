"""Gates JSON em config/."""

import json
from pathlib import Path

from scripts.operations.qa.common import skip


def _json_files(root: Path) -> list[Path]:
    config = root / "config"
    if not config.is_dir():
        return []
    return sorted(path for path in config.rglob("*.json") if path.is_file())


def run_json(stage: str, root: Path) -> None:
    """Executa um estagio de validacao JSON de config/."""
    files = _json_files(root)
    if not files:
        skip("python", "nenhum JSON em config/")
        return
    if stage == "lint":
        for path in files:
            json.loads(path.read_text(encoding="utf-8"))
            print(f"[python] json lint ok {path.relative_to(root)}")
        return
    if stage == "validate":
        settings = root / "config" / "settings.json"
        data = json.loads(settings.read_text(encoding="utf-8"))
        if "symbols" not in data or "deep_learning" not in data:
            raise ValueError("config/settings.json sem symbols ou deep_learning")
        print("[python] json validate ok settings.json")
        return
    if stage == "test":
        python_cfg = root / "config" / "python.json"
        data = json.loads(python_cfg.read_text(encoding="utf-8"))
        name = data.get("conda_env")
        if not isinstance(name, str) or not name:
            raise ValueError("config/python.json sem conda_env")
        print("[python] json test ok python.json")
        return
    skip("python", f"json estagio {stage} coberto por gitleaks")
