import argparse
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


def _project_python_candidates() -> list[Path]:
    if os.name == "nt":
        venv_names = (".venv-win", ".venv")
        layouts = (("Scripts", "python.exe"),)
    else:
        venv_names = (".venv-wsl", ".venv")
        layouts = (("bin", "python"),)
    roots = (APP_ROOT, REPO_ROOT)
    candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for name in venv_names:
            for folder, exe in layouts:
                path = root / name / folder / exe
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    candidates.append(path)
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
    left_key = os.path.normcase(str(left))
    right_key = os.path.normcase(str(right))
    return left_key == right_key


def _ensure_project_python(stage: str) -> None:
    modules = _STAGE_MODULES.get(stage, ())
    current = Path(sys.executable)
    if _imports_available(current, modules):
        return
    for candidate in _project_python_candidates():
        try:
            candidate_exists = candidate.exists()
        except OSError:
            continue
        if not candidate_exists:
            continue
        if _same_interpreter(candidate, current):
            continue
        if _imports_available(candidate, modules):
            completed = subprocess.run([str(candidate), *sys.argv], check=False)  # nosec B603
            sys.exit(completed.returncode)
    if modules:
        missing = ", ".join(modules)
        print(f"\n[ERRO] Dependencias ausentes para o estagio '{stage}': {missing}")
        print("Instale com: make install")
        print(f"Venvs esperados: {APP_ROOT / '.venv-wsl'} (WSL) ou {APP_ROOT / '.venv-win'} (Windows)")
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
        if (
            ".venv" in path.parts
            or "venv" in path.parts
            or ".git" in path.parts
            or ".venv-win" in path.parts
            or ".venv-wsl" in path.parts
        ):
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
    run_tool("coverage", ["run", "-m", "pytest"], "Pytest execution")
    run_tool("coverage", ["report", f"--fail-under={fail_under}"], f"Coverage report (min {fail_under}%)")


def stage_security():
    ignored_vulns = [
        "PYSEC-2022-42969",
        "CVE-2026-45409",
        "CVE-2026-3219",
        "CVE-2026-6357",
        "PYSEC-2025-205",
        "PYSEC-2025-206",
        "PYSEC-2025-200",
        "PYSEC-2025-207",
        "PYSEC-2025-201",
        "PYSEC-2025-204",
        "PYSEC-2026-139",
        "PYSEC-2025-209",
        "PYSEC-2025-208",
        "PYSEC-2025-191",
        "PYSEC-2025-199",
        "PYSEC-2025-202",
        "PYSEC-2025-198",
        "PYSEC-2025-203",
        "CVE-2025-3730",
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
            dirs[:] = [
                d for d in dirs if d not in (".venv", "venv", ".venv-win", ".venv-wsl", ".git", ".idea", ".vscode")
            ]

            for d in list(dirs):
                if d == "__pycache__":
                    safe_remove(Path(root) / d)
                    dirs.remove(d)

            for f in files:
                if f.endswith((".pyc", ".pyo", ".pyd")):
                    safe_remove(Path(root) / f)

    for name in ("data", "logs"):
        p = REPO_ROOT / name
        if p.exists():
            safe_remove(p)


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
