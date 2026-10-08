"""Atalho na raiz para o treino offline do indicador."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Executa o treino com o mesmo interpretador Python."""
    root = Path(__file__).resolve().parent
    raise SystemExit(subprocess.call([sys.executable, str(root / "app" / "train.py")], cwd=root))


if __name__ == "__main__":
    main()
