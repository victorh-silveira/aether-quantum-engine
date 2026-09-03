import argparse
import compileall
import ctypes
import gc
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from scripts.operations.qa.common import AREAS
from scripts.operations.qa.dispatch import run_area_stage, run_config_text

RemoveFn = Callable[[Path], None]
PRESERVED_DATA_CHILDREN = frozenset({"deriv", "dl"})
DOCKER_BIND_MOUNTS_RELATIVE = (Path("infra") / "docker" / "meta-models",)
_CACHE_NAMES = (".pytest_cache", ".ruff_cache", ".coverage", "htmlcov", "dist", "build", ".mypy_cache")
_SKIP_WALK_DIRS = frozenset({".venv", "venv", ".git", ".idea", ".vscode"})
_STAGE_MODULES: dict[str, tuple[str, ...]] = {
    "lint": ("ruff", "interrogate", "vulture"),
    "test": ("torch", "coverage", "pytest", "xdist", "pytest_cov"),
    "pytest": ("torch", "coverage", "pytest", "xdist", "pytest_cov"),
    "security": ("bandit", "pip_audit"),
    "validate": (),
    "build": (),
    "clean": (),
}
_VERBOSE = os.environ.get("AETHER_TEST_VERBOSE", "0") == "1"
_PYTEST_COMMON_ARGS = ("-q", "--timeout=90", "--tb=short", "-p", "no:cacheprovider", "-p", "no:stepwise")


@dataclass(frozen=True)
class TestExecutionProfile:
    name: str
    parallel_workers: int
    available_ram_gb: float | None


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def docker_bind_mount_roots(repo_root: Path) -> tuple[Path, ...]:
    return tuple(repo_root / relative for relative in DOCKER_BIND_MOUNTS_RELATIVE)


def meta_models_root(repo_root: Path) -> Path:
    return repo_root / DOCKER_BIND_MOUNTS_RELATIVE[0]


def is_docker_bind_mount(path: Path, repo_root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return False
    for mount in DOCKER_BIND_MOUNTS_RELATIVE:
        mount_posix = mount.as_posix()
        if relative == mount_posix or relative.startswith(f"{mount_posix}/"):
            return True
    return False


def build_safe_remove(repo_root: Path):
    preserved = docker_bind_mount_roots(repo_root)

    def safe_remove(path: Path) -> None:
        if is_docker_bind_mount(path, repo_root):
            print(f"Preservado bind mount Docker: {path}")
            return
        try:
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removido diretório: {path}")
            else:
                path.unlink()
                print(f"Removido arquivo: {path}")
        except Exception as exc:
            print(f"Erro ao remover {path}: {exc}")

    return safe_remove, preserved


def clean_repo_data(data_root: Path, safe_remove: RemoveFn) -> None:
    if not data_root.exists():
        return
    for child in list(data_root.iterdir()):
        if child.name in PRESERVED_DATA_CHILDREN:
            continue
        safe_remove(child)


def clean_runtime_artifacts(repo_root: Path, safe_remove: RemoveFn) -> None:
    clean_repo_data(repo_root / "data", safe_remove)


def _available_ram_gb() -> float | None:
    result: float | None = None
    if sys.platform == "win32":
        try:
            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                result = stat.ullAvailPhys / (1024**3)
        except (OSError, AttributeError):
            result = None
    elif sys.platform == "linux":
        try:
            with Path("/proc/meminfo").open(encoding="utf-8") as meminfo:
                for line in meminfo:
                    if line.startswith("MemAvailable:"):
                        result = int(line.split()[1]) / (1024 * 1024)
                        break
        except OSError:
            result = None
    return result


def resolve_test_execution_profile() -> TestExecutionProfile:
    cpu_cap = os.cpu_count() or 4
    avail = _available_ram_gb()
    if avail is None:
        return TestExecutionProfile("fallback", max(2, min(4, cpu_cap)), None)
    tiers = (
        (6, "ram-6gb", 2),
        (8, "ram-8gb", 4),
        (16, "ram-16gb", 6),
        (32, "ram-32gb", 8),
    )
    for threshold, name, workers in tiers:
        if avail < threshold:
            return TestExecutionProfile(name, min(workers, cpu_cap), avail)
    return TestExecutionProfile("ram-64gb", min(10, cpu_cap), avail)


def format_profile_summary(profile: TestExecutionProfile) -> str:
    ram = "desconhecida" if profile.available_ram_gb is None else f"{profile.available_ram_gb:.1f} GiB"
    return f"perfil={profile.name} | workers={profile.parallel_workers} | RAM={ram}"


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
            for root in (Path(f"/mnt/c/Users/{user}/anaconda3"), Path(f"/mnt/c/Users/{user}/miniconda3")):
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
    return (
        subprocess.run(
            [str(python), "-c", imports], capture_output=True, text=True, check=False, shell=False
        ).returncode
        == 0
    )


def _ensure_project_python(stage: str) -> None:
    modules = _STAGE_MODULES.get(stage, ())
    current = Path(sys.executable)
    if _imports_available(current, modules):
        return
    for candidate in _conda_python_candidates():
        try:
            exists = candidate.exists()
        except OSError:
            continue
        if not exists or os.path.normcase(str(candidate)) == os.path.normcase(str(current)):
            continue
        if _imports_available(candidate, modules):
            sys.exit(subprocess.run([str(candidate), *map(str, sys.argv)], check=False, shell=False).returncode)
    if modules:
        print(f"\n[ERRO] Dependencias ausentes para o estagio '{stage}': {', '.join(modules)}")
        print(f"Ative o Conda e instale com: conda activate {_conda_env_name()} && make app-install")
    sys.exit(1)


def _use_app_cwd() -> None:
    os.chdir(APP_ROOT)


def run_tool(module: str, args: list[str], description: str) -> None:
    print(f"\n>>> Executando: {description}")
    command = [sys.executable, "-m", module, *args]
    if _VERBOSE:
        print(f"Command: {' '.join(command)}")
    subprocess.run(command, check=True, text=True, shell=False)


def _run_subprocess(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    if _VERBOSE:
        subprocess.run(command, check=True, text=True, shell=False, cwd=cwd, env=env)
        return
    completed = subprocess.run(command, check=False, text=True, shell=False, cwd=cwd, env=env, capture_output=True)
    if completed.returncode == 0:
        return
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    raise subprocess.CalledProcessError(completed.returncode, command, output=completed.stdout, stderr=completed.stderr)


def _purge_coverage_artifacts(app_root: Path) -> None:
    for pattern in (".coverage", ".coverage.*"):
        for artifact in app_root.glob(pattern):
            artifact.unlink(missing_ok=True)


def _release_parent_memory() -> None:
    gc.collect()
    if sys.platform != "linux":
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        return


def clean_named_caches(scan_root: Path, safe_remove: RemoveFn) -> None:
    for name in _CACHE_NAMES:
        candidate = scan_root / name
        if candidate.exists():
            safe_remove(candidate)


def clean_python_artifacts(scan_root: Path, safe_remove: RemoveFn, repo_root: Path) -> None:
    for root, dirs, files in os.walk(scan_root):
        dirs[:] = [
            entry
            for entry in dirs
            if entry not in _SKIP_WALK_DIRS and not is_docker_bind_mount(Path(root) / entry, repo_root)
        ]
        for entry in list(dirs):
            if entry == "__pycache__":
                safe_remove(Path(root) / entry)
                dirs.remove(entry)
        for filename in files:
            if filename.endswith((".pyc", ".pyo", ".pyd")):
                safe_remove(Path(root) / filename)


def clean_workspace_artifacts(app_root: Path, repo_root: Path, safe_remove: RemoveFn) -> None:
    for scan_root in (app_root, repo_root):
        clean_named_caches(scan_root, safe_remove)
        clean_python_artifacts(scan_root, safe_remove, repo_root)
    logs_dir = repo_root / "logs"
    if logs_dir.exists():
        safe_remove(logs_dir)
    app_data = app_root / "data"
    if app_data.exists():
        safe_remove(app_data)
    for pattern in (repo_root.glob("pytest-cache-files-*"), app_root.glob("pytest-cache-files-*")):
        for stray in pattern:
            safe_remove(stray)


def stage_clean(*, light: bool = False) -> None:
    print("\n>>> Executando: Limpeza de lixo e caches")
    safe_remove, preserved_mounts = build_safe_remove(REPO_ROOT)
    for mount in preserved_mounts:
        print(f"Bind mount Docker preservado: {mount}")
    clean_workspace_artifacts(APP_ROOT, REPO_ROOT, safe_remove)
    if light:
        return
    print("\n>>> Executando: Limpeza de dados locais de run/treino")
    clean_runtime_artifacts(REPO_ROOT, safe_remove)


def stage_structure(max_lines: int = 300) -> None:
    print(f"\n>>> Executando: Verificação Estrutural (Max {max_lines} linhas)")
    violations = []
    src_root = APP_ROOT / "src"
    for path in src_root.rglob("*.py"):
        count = len(path.read_text(encoding="utf-8").splitlines())
        if count > max_lines:
            violations.append(f"{path}: {count} linhas")
    if violations:
        print("\n[ERRO] Violação de limite de linhas encontrada:")
        for violation in violations:
            print(f"  - {violation}")
        sys.exit(1)
    print(f"[OK] Todos os arquivos estão abaixo de {max_lines} linhas.")


def stage_lint() -> None:
    print("\n>>> Executando: Ruff Check (auto-fix)")
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", "."], check=True, text=True, shell=False)
    run_tool("ruff", ["format", "."], "Ruff Format")
    run_tool("ruff", ["check", "."], "Ruff Check")
    run_tool("interrogate", ["-vv", "src"], "Interrogate Docstrings")
    run_tool("vulture", [], "Vulture Dead Code Detection")
    stage_structure()


def _test_env_base() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("COVERAGE_CORE", "sysmon")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("PYTORCH_NUM_THREADS", "1")
    return env


def stage_test(fail_under: int = 100) -> None:
    profile = resolve_test_execution_profile()
    workers = profile.parallel_workers
    print(f"\n>>> {format_profile_summary(profile)}")
    _purge_coverage_artifacts(APP_ROOT)
    env = _test_env_base()
    command = [
        sys.executable,
        "-m",
        "pytest",
        *_PYTEST_COMMON_ARGS,
        "--import-mode=importlib",
        "-n",
        str(workers),
        "--cov=src",
        "--cov-report=term-missing",
        f"--cov-fail-under={fail_under}",
        "tests/",
    ]
    print(f">>> pytest-xdist + pytest-cov ({workers} workers)")
    _run_subprocess(command, cwd=APP_ROOT, env=env)
    _release_parent_memory()


def _resolve_gitleaks_bin() -> str | None:
    found = shutil.which("gitleaks")
    if found:
        return found
    for candidate in ("/usr/bin/gitleaks", "/usr/local/bin/gitleaks"):
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _repo_root_for_bash(path: Path) -> str:
    posix = path.resolve().as_posix()
    if len(posix) >= 2 and posix[1] == ":":
        drive = posix[0].lower()
        return f"/mnt/{drive}{posix[2:]}"
    return posix


def _run_gitleaks() -> None:
    print("\n>>> Executando: Gitleaks Secret Scan")
    source = _repo_root_for_bash(REPO_ROOT)
    args = ["detect", "--source", source, "--verbose", "--redact"]
    commands: list[list[str]] = []
    binary = _resolve_gitleaks_bin()
    if binary is not None:
        commands.append([binary, *args])
    bash = shutil.which("bash")
    if bash is not None:
        commands.append(
            [
                bash,
                "-lc",
                f'gitleaks detect --source "{source}" --verbose --redact',
            ]
        )
    for command in commands:
        try:
            res = subprocess.run(command, check=False, cwd=str(REPO_ROOT), text=True, shell=False)
            if res.returncode == 0:
                return
        except (FileNotFoundError, OSError):
            continue
    if str(os.environ.get("CI", "")).strip().lower() in {"1", "true", "yes"}:
        print("[ERRO] gitleaks obrigatorio no CI")
        sys.exit(1)
    print("[AVISO] gitleaks nao encontrado no PATH local ou retornou codigo diferente de 0. Prosseguindo...")
    return


def stage_python_compileall(*, quiet: bool) -> None:
    target = APP_ROOT / "src"
    print("\n>>> Executando: compileall app/src")
    ok = compileall.compile_dir(str(target), quiet=1 if quiet else 0, force=False)
    if not ok:
        sys.exit(1)


def stage_security() -> None:
    run_tool("bandit", ["-r", "src", "-c", "pyproject.toml"], "Bandit Security Scan")
    ignored_vulns = ["PYSEC-2022-42969", "PYSEC-2026-139", "CVE-2025-3000", "PYSEC-2026-3447"]
    ignore_args = ["--local"] + [item for vuln in ignored_vulns for item in ("--ignore-vuln", vuln)]
    try:
        run_tool("pip_audit", ignore_args, "Pip-audit Vulnerability Scan")
    except subprocess.CalledProcessError:
        print("[AVISO] Pip-audit encontrou vulnerabilidades no ambiente global de pacotes Python.")
    _run_gitleaks()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aether Engine Quality Gate")
    parser.add_argument(
        "--area",
        default="python",
        choices=list(AREAS),
        help="Area da matriz CI",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["lint", "pytest", "security", "test", "clean", "validate", "build"],
        help="Stage to execute",
    )
    parser.add_argument("--coverage-fail-under", type=int, default=100, help="Minimum coverage percentage")
    parser.add_argument(
        "--light-clean", action="store_true", help="Limpa apenas caches sem remover dados de run/treino"
    )
    parser.add_argument(
        "--config-text",
        nargs="?",
        const="all",
        choices=["all", "json", "yaml"],
        help="Valida JSON e/ou YAML no job Python (nao e stack)",
    )
    args = parser.parse_args()
    if args.config_text is not None:
        if args.area != "python":
            parser.error("--config-text exige --area python")
        run_config_text(str(args.stage), REPO_ROOT, kind=str(args.config_text))
        print("\n[SUCESSO] Estágio concluído com sucesso.")
        return
    if args.stage == "clean" or args.area == "python":
        _ensure_project_python(args.stage)
        _use_app_cwd()
        if args.stage == "lint":
            stage_lint()
        elif args.stage in ("pytest", "test"):
            stage_test(args.coverage_fail_under)
        elif args.stage == "security":
            stage_security()
        elif args.stage == "validate":
            stage_python_compileall(quiet=False)
        elif args.stage == "build":
            stage_python_compileall(quiet=True)
        elif args.stage == "clean":
            stage_clean(light=args.light_clean)
        print("\n[SUCESSO] Estágio concluído com sucesso.")
        return
    run_area_stage(str(args.area), str(args.stage), REPO_ROOT)
    print("\n[SUCESSO] Estágio concluído com sucesso.")


if __name__ == "__main__":
    main()
