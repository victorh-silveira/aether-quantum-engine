"""Gates YAML (compose, pre-commit, GitHub Actions)."""

from pathlib import Path

from scripts.operations.qa.common import require_tool, run_cmd, skip


def _yaml_files(root: Path) -> list[Path]:
    candidates = [
        root / "infra" / "docker" / "docker-compose.yml",
        root / ".pre-commit-config.yaml",
    ]
    github = root / ".github"
    if github.is_dir():
        candidates.extend(sorted(github.rglob("*.yml")))
        candidates.extend(sorted(github.rglob("*.yaml")))
    return [path for path in candidates if path.is_file()]


def _assert_yaml_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{path} vazio")
    for line in text.splitlines():
        if line.startswith("\t"):
            raise ValueError(f"{path} usa tab na indentacao")


def run_yaml(stage: str, root: Path) -> None:
    """Executa um estagio de validacao YAML operacional."""
    files = _yaml_files(root)
    if not files:
        skip("python", "nenhum YAML")
        return
    if stage == "lint":
        for path in files:
            _assert_yaml_text(path)
            print(f"[python] yaml lint ok {path.relative_to(root)}")
        return
    if stage == "validate":
        workflows = root / ".github" / "workflows"
        if not workflows.is_dir():
            skip("python", "workflows ausentes")
            return
        workflow_files = sorted(path for path in workflows.glob("*.yml") if path.is_file()) + sorted(
            path for path in workflows.glob("*.yaml") if path.is_file()
        )
        if not workflow_files:
            skip("python", "nenhum workflow yml")
            return
        actionlint = require_tool("actionlint", area="python")
        if actionlint is None:
            return
        run_cmd(
            [actionlint, *[str(path) for path in workflow_files]],
            cwd=root,
            description="Actionlint workflows",
        )
        return
    skip("python", f"yaml estagio {stage} coberto por gitleaks")
