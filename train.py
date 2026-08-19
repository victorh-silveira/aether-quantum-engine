import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent
    target = repo / "app" / "scripts" / "operations" / "run_launch_train_tf_pipeline.py"
    args = [sys.executable, str(target), *sys.argv[1:]]
    raise SystemExit(subprocess.call(args, cwd=repo))


if __name__ == "__main__":
    main()
