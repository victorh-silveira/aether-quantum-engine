import argparse
import ctypes
import gc
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

RemoveFn = Callable[[Path], None]
PRESERVED_DATA_CHILDREN = frozenset({"deriv"})
DOCKER_BIND_MOUNTS_RELATIVE = (
    Path("infra") / "docker" / "triton-models",
    Path("infra") / "docker" / "meta-models",
)
_CACHE_NAMES = (".pytest_cache", ".ruff_cache", ".coverage", "htmlcov", "dist", "build", ".mypy_cache")
_SKIP_WALK_DIRS = frozenset({".venv", "venv", ".git", ".idea", ".vscode"})
_STAGE_MODULES: dict[str, tuple[str, ...]] = {
    "lint": ("ruff", "interrogate", "vulture", "pylint"),
    "test": ("torch", "coverage", "pytest"),
    "pytest": ("torch", "coverage", "pytest"),
    "security": ("bandit", "pip_audit"),
    "clean": (),
}
_TORCH_GC_BATCH_SIZE = "3"
_VERBOSE = os.environ.get("AETHER_TEST_VERBOSE", "0") == "1"
_PYTEST_COMMON_ARGS = ("-q", "--timeout=90", "--tb=short", "-p", "no:cacheprovider", "-p", "no:stepwise")
_PYTEST_LIGHT_ARGS = (*_PYTEST_COMMON_ARGS, "--import-mode=prepend")
_PYTEST_HEAVY_ARGS = (*_PYTEST_COMMON_ARGS, "--import-mode=importlib")
_HEAVY_NAME_MARKERS = (
    "torch",
    "deep_learning",
    "dl_",
    "stream_handler",
    "websocket",
    "triton",
    "train_meta",
    "infra_coverage_extended",
    "post_settlement",
    "minio_torch",
)
_ISOLATED_MODULES = frozenset(
    {
        "tests/unit/infrastructure/test_torchscript_sanity.py",
        "tests/unit/infrastructure/test_minio_torchscript_sanity.py",
        "tests/unit/infrastructure/test_infra_coverage_extended.py",
        "tests/unit/infrastructure/test_stream_handler.py",
        "tests/unit/infrastructure/test_triton_grpc_client.py",
        "tests/unit/infrastructure/test_websocket_manager.py",
        "tests/unit/application/test_dl_coverage_gaps.py",
        "tests/unit/application/test_deep_learning.py",
        "tests/unit/application/test_dl_device.py",
        "tests/unit/application/test_post_settlement_cycle.py",
        "tests/unit/scripts/test_train_meta_pipeline.py",
    }
)
_DEFAULT_SHARD_FILE_LIMIT = 8


@dataclass(frozen=True)
class TestExecutionProfile:
    name: str
    shard_files: int
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


def triton_models_root(repo_root: Path) -> Path:
    return repo_root / DOCKER_BIND_MOUNTS_RELATIVE[0]


def meta_models_root(repo_root: Path) -> Path:
    return repo_root / DOCKER_BIND_MOUNTS_RELATIVE[1]


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
    mode = os.environ.get("AETHER_TEST_PROFILE", "auto").strip().lower()
    if mode == "safe":
        return TestExecutionProfile("safe", 4, 1, avail)
    if mode == "fast":
        workers = min(6, cpu_cap)
        if avail is not None:
            workers = max(2, min(workers, int(avail // 1.5) or 2))
        return TestExecutionProfile("fast", 16, workers, avail)
    if avail is None:
        return TestExecutionProfile("safe-fallback", 6, max(1, min(2, cpu_cap)), None)
    tiers = (
        (8, "ram-8gb", 8, 2),
        (16, "ram-16gb", 10, 3),
        (32, "ram-32gb", 14, 5),
    )
    for threshold, name, shard_files, workers in tiers:
        if avail < threshold:
            return TestExecutionProfile(name, shard_files, min(workers, cpu_cap), avail)
    return TestExecutionProfile("ram-64gb", 18, min(8, cpu_cap), avail)


def profile_shard_files(profile: TestExecutionProfile) -> int:
    raw = os.environ.get("AETHER_TEST_SHARD_FILES")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return profile.shard_files
    return profile.shard_files


def profile_worker_count(profile: TestExecutionProfile) -> int:
    raw = os.environ.get("AETHER_TEST_SHARD_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return profile.parallel_workers
    return profile.parallel_workers


def format_profile_summary(profile: TestExecutionProfile) -> str:
    ram = "desconhecida" if profile.available_ram_gb is None else f"{profile.available_ram_gb:.1f} GiB"
    return (
        f"perfil={profile.name} | arquivos/shard={profile.shard_files} | workers={profile.parallel_workers} | RAM={ram}"
    )


def _is_heavy_module(path: str) -> bool:
    if path in _ISOLATED_MODULES:
        return True
    name = Path(path).name.lower()
    return any(marker in name for marker in _HEAVY_NAME_MARKERS)


def _collect_all_test_paths(app_root: Path) -> list[str]:
    paths: list[str] = []
    paths.extend(sorted(p.relative_to(app_root).as_posix() for p in app_root.glob("tests/unit/application/test_*.py")))
    paths.extend(sorted(p.relative_to(app_root).as_posix() for p in app_root.glob("tests/unit/domain/**/test_*.py")))
    paths.extend(
        sorted(p.relative_to(app_root).as_posix() for p in app_root.glob("tests/unit/infrastructure/test_*.py"))
    )
    paths.extend(sorted(p.relative_to(app_root).as_posix() for p in app_root.glob("tests/unit/scripts/test_*.py")))
    paths.extend(
        sorted(p.relative_to(app_root).as_posix() for p in app_root.glob("tests/unit/presentation/**/test_*.py"))
    )
    run_test = app_root / "tests/unit/test_run.py"
    if run_test.is_file():
        paths.append(run_test.relative_to(app_root).as_posix())
    integration = app_root / "tests/integration"
    if integration.is_dir():
        paths.extend(sorted(p.relative_to(app_root).as_posix() for p in integration.rglob("test_*.py")))
    return paths


def build_coverage_shards(app_root: Path, shard_files: int | None = None) -> list[list[str]]:
    limit = shard_files if shard_files is not None else _DEFAULT_SHARD_FILE_LIMIT
    paths = _collect_all_test_paths(app_root)
    heavy = [[path] for path in paths if _is_heavy_module(path)]
    light_paths = [path for path in paths if not _is_heavy_module(path)]
    light = [
        light_paths[index : index + limit]
        for index in range(0, len(light_paths), limit)
        if light_paths[index : index + limit]
    ]
    return light + heavy


def split_heavy_light_shards(shards: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    light: list[list[str]] = []
    heavy: list[list[str]] = []
    for shard in shards:
        (heavy if any(_is_heavy_module(target) for target in shard) else light).append(shard)
    return light, heavy


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


def _run_pytest_shard(app_root: Path, *, targets: list[str], shard_index: int, shard_total: int) -> None:
    heavy = any(_is_heavy_module(target) for target in targets)
    pytest_args = _PYTEST_HEAVY_ARGS if heavy else _PYTEST_LIGHT_ARGS
    command = [sys.executable, "-m", "coverage", "run", "-m", "pytest", *pytest_args, *targets]
    env = os.environ.copy()
    env["COVERAGE_FILE"] = f".coverage.{shard_index:04d}"
    env.setdefault("COVERAGE_CORE", "sysmon")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("PYTORCH_NUM_THREADS", "1")
    env.setdefault("AETHER_GC_BATCH_SIZE", _TORCH_GC_BATCH_SIZE if heavy else "15")
    print(f">>> Shard {shard_index}/{shard_total} [{'heavy' if heavy else 'light'}] ({len(targets)} arquivos)")
    _run_subprocess(command, cwd=app_root, env=env)


def _run_shards_parallel(app_root: Path, shards: list[list[str]], *, workers: int, shard_total: int) -> None:
    if not shards:
        return
    pool_workers = min(workers, len(shards))
    print(f">>> Paralelo: {pool_workers} workers em {len(shards)} shards leves")
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        futures = [
            pool.submit(_run_pytest_shard, app_root, targets=targets, shard_index=index, shard_total=shard_total)
            for index, targets in enumerate(shards, start=1)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as exc:
                errors.append(exc)
    if errors:
        raise errors[0]


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
    for path in APP_ROOT.rglob("*.py"):
        if ".venv" in path.parts or "venv" in path.parts or ".git" in path.parts or "scripts" in path.parts:
            continue
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


def stage_test(fail_under: int = 100) -> None:
    profile = resolve_test_execution_profile()
    shard_files = profile_shard_files(profile)
    workers = profile_worker_count(profile)
    print(f"\n>>> {format_profile_summary(profile)}")
    _purge_coverage_artifacts(APP_ROOT)
    shards = build_coverage_shards(APP_ROOT, shard_files=shard_files)
    total = len(shards)
    if total == 0:
        raise RuntimeError("Nenhum alvo de teste encontrado para execucao fragmentada.")
    light_shards, heavy_shards = split_heavy_light_shards(shards)
    print(f">>> Plano: {len(light_shards)} shards leves + {len(heavy_shards)} shards pesados (total {total})")
    _run_shards_parallel(APP_ROOT, light_shards, workers=workers, shard_total=total)
    for index, targets in enumerate(heavy_shards, start=len(light_shards) + 1):
        _run_pytest_shard(APP_ROOT, targets=targets, shard_index=index, shard_total=total)
    _release_parent_memory()
    run_tool("coverage", ["combine"], "Coverage combine")
    run_tool("coverage", ["report", f"--fail-under={fail_under}"], f"Coverage report (min {fail_under}%)")


def stage_security() -> None:
    ignored_vulns = ["PYSEC-2022-42969", "PYSEC-2026-139", "CVE-2025-3000"]
    ignore_args = [item for vuln in ignored_vulns for item in ("--ignore-vuln", vuln)]
    run_tool("bandit", ["-r", ".", "-c", "pyproject.toml"], "Bandit Security Scan")
    run_tool("pip_audit", ignore_args, "Pip-audit Vulnerability Scan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aether Engine Quality Gate")
    parser.add_argument(
        "--stage", required=True, choices=["lint", "pytest", "security", "test", "clean"], help="Stage to execute"
    )
    parser.add_argument("--coverage-fail-under", type=int, default=100, help="Minimum coverage percentage")
    parser.add_argument(
        "--light-clean", action="store_true", help="Limpa apenas caches sem remover dados de run/treino"
    )
    args = parser.parse_args()
    _ensure_project_python(args.stage)
    _use_app_cwd()
    if args.stage == "lint":
        stage_lint()
    elif args.stage in ("pytest", "test"):
        stage_test(args.coverage_fail_under)
    elif args.stage == "security":
        stage_security()
    elif args.stage == "clean":
        stage_clean(light=args.light_clean)
    print("\n[SUCESSO] Estágio concluído com sucesso.")


if __name__ == "__main__":
    main()
