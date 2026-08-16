"""Limpa checkpoints e artefactos de run anterior antes de train/docker-reset.

Nao usar em docker-rebuild pos-treino: isso apaga data/dl e o TCN recem-exportado.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable
from pathlib import Path


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from aether_paths import REPO_ROOT


PRESERVE_DATA_DIRS = frozenset({"deriv"})
DL_GLOBS = ("*.pth", "*_ts.pt", "*.pt")
MODEL_GLOBS = ("*.pkl", "*.joblib")
META_BUNDLE_NAMES = frozenset({"meta_lgbm.pkl"})
RemoveFn = Callable[[Path], None]


def _default_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)


def _clear_glob(root: Path, patterns: tuple[str, ...], remove: RemoveFn) -> int:
    if not root.is_dir():
        root.mkdir(parents=True, exist_ok=True)
        return 0
    removed = 0
    for pattern in patterns:
        for path in root.glob(pattern):
            remove(path)
            removed += 1
    return removed


def clear_dl_checkpoints(repo_root: Path, remove: RemoveFn) -> int:
    return _clear_glob(repo_root / "data" / "dl", DL_GLOBS, remove)


def clear_repo_data_runtime(repo_root: Path, remove: RemoveFn) -> int:
    data_root = repo_root / "data"
    if not data_root.is_dir():
        return 0
    removed = 0
    for child in list(data_root.iterdir()):
        if child.name in PRESERVE_DATA_DIRS or child.name == "dl":
            continue
        remove(child)
        removed += 1
    return removed


def clear_bind_model_dir(
    path: Path,
    remove: RemoveFn,
    *,
    keep_names: frozenset[str] | None = None,
) -> int:
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        return 0
    removed = 0
    keep = keep_names or frozenset()
    for pattern in MODEL_GLOBS:
        for candidate in path.glob(pattern):
            if candidate.name in keep:
                continue
            remove(candidate)
            removed += 1
    return removed


def sanitize_fresh_run(
    repo_root: Path,
    *,
    remove: RemoveFn | None = None,
    keep_meta_bundle: bool = False,
) -> dict[str, int]:
    """Remove checkpoints DL, pkls meta/loss e estado em data/ (exceto deriv)."""
    rem = remove or _default_remove
    meta_keep = META_BUNDLE_NAMES if keep_meta_bundle else frozenset()
    return {
        "dl": clear_dl_checkpoints(repo_root, rem),
        "data_runtime": clear_repo_data_runtime(repo_root, rem),
        "meta": clear_bind_model_dir(
            repo_root / "infra" / "docker" / "meta-models",
            rem,
            keep_names=meta_keep,
        ),
        "loss": clear_bind_model_dir(repo_root / "infra" / "docker" / "loss-models", rem),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sanitiza run anterior (checkpoints e artefactos).")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Raiz do repositorio (default: monorepo).",
    )
    parser.add_argument(
        "--keep-meta-bundle",
        action="store_true",
        help="Preserva meta_lgbm.pkl (docker-reset antes do train).",
    )
    args = parser.parse_args(argv)
    counts = sanitize_fresh_run(
        args.repo_root.resolve(),
        keep_meta_bundle=bool(args.keep_meta_bundle),
    )
    total = sum(counts.values())
    print(
        f"[AETHER] sanitize ok dl={counts['dl']} data={counts['data_runtime']} "
        f"meta={counts['meta']} loss={counts['loss']} total={total}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
