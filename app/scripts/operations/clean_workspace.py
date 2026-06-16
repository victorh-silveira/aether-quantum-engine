import argparse
import json
import os
import shutil
import subprocess  # nosec
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_ROOT.parent

_STAGE_MODULES: dict[str, tuple[str, ...]] = {
    "lint": ("ruff", "interrogate", "vulture", "pylint"),
    "test": ("torch", "coverage", "pytest"),
    "pytest": ("torch", "coverage", "pytest"),
    "security": ("bandit", "pip_audit"),
    "clean": (),
}


def _conda_env_name() -> str:
    override = os.environ.get("AETHER_CONDA_ENV")
    if override:
        return override
    cfg = REPO_ROOT / "config" / "python.json"
    if cfg.is_file():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        name = data.get("conda_env")
        if isinstance(name, str) and name:
            return name
    return "deriv-api"


def _conda_python_candidates() -> list[Path]:
    env_name = _conda_env_name()
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and Path(prefix).name == env_name:
        add(Path(prefix) / "python.exe")
        add(Path(prefix) / "bin" / "python")

    home = Path.home()
    for root in (home / "anaconda3", home / "miniconda3", Path("C:/ProgramData/anaconda3")):
        add(root / "envs" / env_name / "python.exe")
        add(root / "envs" / env_name / "bin" / "python")

    if os.name != "nt":
        user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
        if user:
            for root in (
                Path(f"/mnt/c/Users/{user}/anaconda3"),
                Path(f"/mnt/c/Users/{user}/miniconda3"),
            ):
                add(root / "envs" / env_name / "python.exe")
                add(root / "envs" / env_name / "bin" / "python")

    return candidates


def _imports_available(python: Path, modules: tuple[str, ...]) -> bool:
    if not modules:
        return True
    try:
        if not python.exists():
            return False
    except OSError:
        return False
    imports = "; ".join(f"import {module}" for module in modules)
    result = subprocess.run(
        [str(python), "-c", imports],
        capture_output=True,
        text=True,
        check=False,
    )  # nosec B603
    return result.returncode == 0


def _same_interpreter(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _ensure_project_python(stage: str) -> None:
    modules = _STAGE_MODULES.get(stage, ())
    current = Path(sys.executable)
    if _imports_available(current, modules):
        return
    env_name = _conda_env_name()
    for candidate in _conda_python_candidates():
        try:
            exists = candidate.exists()
        except OSError:
            continue
        if not exists or _same_interpreter(candidate, current):
            continue
        if _imports_available(candidate, modules):
            completed = subprocess.run([str(candidate), *sys.argv], check=False)  # nosec B603
            sys.exit(completed.returncode)
    if modules:
        missing = ", ".join(modules)
        print(f"\n[ERRO] Dependencias ausentes para o estagio '{stage}': {missing}")
        print(f"Ative o Conda e instale com: conda activate {env_name} && make install")
    sys.exit(1)


def _use_app_cwd() -> None:
    os.chdir(APP_ROOT)


def run_tool(module, args, description):
    print(f"\n>>> Executando: {description}")
    command = [sys.executable, "-m", module] + args
    print(f"Command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, text=True)  # nosec
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erro durante {description}: {e}")
        sys.exit(e.returncode)


def stage_lint():
    print("\n>>> Executando: Ruff Check (auto-fix)")
    fix_cmd = [sys.executable, "-m", "ruff", "check", "--fix", "."]
    print(f"Command: {' '.join(fix_cmd)}")
    subprocess.run(fix_cmd, check=True, text=True)  # nosec
    run_tool("ruff", ["check", "."], "Ruff Check")
    run_tool("ruff", ["format", "."], "Ruff Format")
    run_tool("interrogate", ["-vv", "."], "Interrogate Docstrings")
    run_tool("vulture", [], "Vulture Dead Code Detection")
    run_tool(
        "pylint",
        ["--disable=all", "--enable=duplicate-code", "--min-similarity-lines=15", "src/"],
        "Pylint Duplicate Code Detection",
    )
    stage_structure()


def stage_structure(max_lines=300):
    print(f"\n>>> Executando: Verificação Estrutural (Max {max_lines} linhas)")
    violations = []

    for path in APP_ROOT.rglob("*.py"):
        if ".venv" in path.parts or "venv" in path.parts or ".git" in path.parts:
            continue

        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            count = len(lines)
            if count > max_lines:
                violations.append(f"{path}: {count} linhas")

    if violations:
        print("\n[ERRO] Violação de limite de linhas encontrada:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    print(f"[OK] Todos os arquivos estão abaixo de {max_lines} linhas.")


def stage_test(fail_under=100):
    data_file = APP_ROOT / ".coverage"
    if data_file.exists():
        data_file.unlink()
    run_tool("coverage", ["run", "-m", "pytest", "--timeout=90"], "Pytest execution")
    run_tool("coverage", ["report", f"--fail-under={fail_under}"], f"Coverage report (min {fail_under}%)")


def stage_security():
    ignored_vulns = [
        "PYSEC-2022-42969",
        "PYSEC-2026-139",
        "CVE-2025-3000",
    ]
    ignore_args = []
    for vuln in ignored_vulns:
        ignore_args.extend(["--ignore-vuln", vuln])

    run_tool("bandit", ["-r", ".", "-c", "pyproject.toml"], "Bandit Security Scan")
    run_tool("pip_audit", ignore_args, "Pip-audit Vulnerability Scan")


def stage_clean():
    print("\n>>> Running: Limpeza de lixo e caches")

    def safe_remove(p: Path):
        try:
            if p.is_dir():
                shutil.rmtree(p)
                print(f"Removido diretório: {p}")
            else:
                p.unlink()
                print(f"Removido arquivo: {p}")
        except Exception as e:
            print(f"Erro ao remover {p}: {e}")

    cache_names = (
        ".pytest_cache",
        ".ruff_cache",
        ".coverage",
        "htmlcov",
        "dist",
        "build",
        ".mypy_cache",
    )

    for scan_root in (APP_ROOT, REPO_ROOT):
        # 1. Remover caches comuns no topo
        for name in cache_names:
            p = scan_root / name
            if p.exists():
                safe_remove(p)

        # 2. Varredura inteligente de __pycache__ e bytecodes
        for root, dirs, files in os.walk(scan_root):
            # Ignora pastas pesadas ou do ambiente virtual
            dirs[:] = [d for d in dirs if d not in (".venv", "venv", ".git", ".idea", ".vscode")]

            for d in list(dirs):
                if d == "__pycache__":
                    safe_remove(Path(root) / d)
                    dirs.remove(d)

            for f in files:
                if f.endswith((".pyc", ".pyo", ".pyd")):
                    safe_remove(Path(root) / f)

    for name in ("logs",):
        p = REPO_ROOT / name
        if p.exists():
            safe_remove(p)

    app_data = APP_ROOT / "data"
    if app_data.exists():
        safe_remove(app_data)

    for stray in APP_ROOT.glob("pytest-cache-files-*"):
        safe_remove(stray)


def main():
    parser = argparse.ArgumentParser(description="Aether Engine Quality Gate")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["lint", "pytest", "security", "test", "clean"],
        help="Stage to execute",
    )
    parser.add_argument("--coverage-fail-under", type=int, default=100, help="Minimum coverage percentage")

    args = parser.parse_args()
    _ensure_project_python(args.stage)
    _use_app_cwd()

    if args.stage == "lint":
        stage_lint()
    elif args.stage in ["pytest", "test"]:
        stage_test(args.coverage_fail_under)
    elif args.stage == "security":
        stage_security()
    elif args.stage == "clean":
        stage_clean()

    print("\n[SUCESSO] Estágio concluído com sucesso.")


if __name__ == "__main__":
    main()
