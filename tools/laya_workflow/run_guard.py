"""Exclusao mutua de treinamentos e protecao contra execucoes legadas ativas."""

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path


def other_training_pids() -> list[int]:
    """Consulta somente PIDs de outros treinadores sem expor argumentos de credenciais."""
    if os.name == "nt":
        script = (
            "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\" | "
            f"Where-Object {{ $_.ProcessId -ne {os.getpid()} -and "
            "$_.CommandLine -match 'tools[.]laya_workflow' -and $_.CommandLine -match ' train(?: |$)' } | "
            "ForEach-Object { $_.ProcessId }"
        )
        executable = Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        result = subprocess.run(  # noqa: S603
            [str(executable), "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = path.read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if int(path.parent.name) != os.getpid() and b"tools.laya_workflow" in args and b"train" in args:
            found.append(int(path.parent.name))
    return found


@contextmanager
def training_lock(directory: Path):
    """Serializa os treinadores cooperantes e recusa um legado ainda em execucao."""
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".training.lock"
    with lock_path.open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl  # noqa: PLC0415

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            others = other_training_pids()
            if others:
                raise RuntimeError(f"Outro treinamento ativo (PIDs {others}); encerre-o antes de iniciar esta execucao")
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
