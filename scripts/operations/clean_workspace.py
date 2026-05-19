import argparse
import shutil
import subprocess  # nosec
import sys
from pathlib import Path


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
    run_tool("ruff", ["check", "--fix", "--exit-non-zero-on-fix", "."], "Ruff Check")
    run_tool("ruff", ["format", "."], "Ruff Format")
    run_tool("interrogate", ["-vv", "."], "Interrogate Docstrings")
    run_tool("vulture", [], "Vulture Dead Code Detection")
    stage_structure()


def stage_structure(max_lines=300):
    print(f"\n>>> Executando: Verificação Estrutural (Max {max_lines} linhas)")
    root = Path()
    violations = []

    for path in root.rglob("*.py"):
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
    run_tool("coverage", ["run", "-m", "pytest"], "Pytest execution")
    run_tool("coverage", ["report", f"--fail-under={fail_under}"], f"Coverage report (min {fail_under}%)")


def stage_security():
    ignored_vulns = ["PYSEC-2022-42969", "CVE-2026-45409"]
    ignore_args = []
    for vuln in ignored_vulns:
        ignore_args.extend(["--ignore-vuln", vuln])

    run_tool("bandit", ["-r", ".", "-c", "pyproject.toml"], "Bandit Security Scan")
    run_tool("pip_audit", ignore_args, "Pip-audit Vulnerability Scan")


def stage_clean():
    print("\n>>> Running: Limpeza de lixo e caches")
    root = Path()

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

    for path in root.rglob("__pycache__"):
        if path.is_dir():
            safe_remove(path)

    for ext in ("*.pyc", "*.pyo", "*.pyd"):
        for path in root.rglob(ext):
            if path.is_file():
                safe_remove(path)

    for name in (
        ".pytest_cache",
        ".ruff_cache",
        ".coverage",
        "htmlcov",
        "data",
        "logs",
        "dist",
        "build",
    ):
        p = root / name
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
