"""Gates de scripts bash."""

from pathlib import Path

from scripts.operations.qa.common import posix_for_tool, require_tool, run_cmd, skip, which


def _shell_files(root: Path) -> list[Path]:
    globs = (
        root / "infra" / "docker",
        root / "linters" / "git-hooks",
        root / "app" / "scripts" / "wsl",
    )
    found: list[Path] = []
    for folder in globs:
        if not folder.is_dir():
            continue
        found.extend(sorted(folder.rglob("*.sh")))
    return found


def run_shell(stage: str, root: Path) -> None:
    """Executa um estagio da area shell."""
    files = _shell_files(root)
    if not files:
        skip("shell", "nenhum script .sh")
        return
    if stage == "lint":
        shellcheck = which("shellcheck")
        if shellcheck:
            for path in files:
                run_cmd(
                    [shellcheck, "-e", "SC1091", posix_for_tool(path)],
                    cwd=root,
                    description=f"shellcheck {path.relative_to(root)}",
                )
            return
        bash = require_tool("bash", area="shell")
        if bash is None:
            return
        for path in files:
            run_cmd(
                [bash, "-n", posix_for_tool(path)],
                cwd=root,
                description=f"bash -n {path.relative_to(root)}",
            )
        return
    if stage == "validate":
        bash = require_tool("bash", area="shell")
        if bash is None:
            return
        for path in files:
            run_cmd(
                [bash, "-n", posix_for_tool(path)],
                cwd=root,
                description=f"bash -n {path.relative_to(root)}",
            )
        return
    skip("shell", f"estagio {stage} nao aplicavel")
