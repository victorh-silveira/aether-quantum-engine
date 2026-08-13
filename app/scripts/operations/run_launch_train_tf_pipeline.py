"""Pipeline launch-train: sweep multi-TF + promote do vencedor (fail-closed)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from aether_paths import REPO_ROOT
from scripts.operations.promote_tf_winner import main as promote_main
from scripts.operations.sweep_train_timeframes import run_tf_sweep
from src.application.services.deep_learning.tf_sweep_config import (
    load_tf_sweep_knobs,
    resolve_repo_path,
)
from src.application.services.deep_learning.tf_sweep_score import pick_tf_winner
from src.domain.config_knobs import load_settings_json
from src.presentation.terminal.logger import setup_logger


def clear_sweep_artifacts(artifact_root: str, *, repo_root: Path | None = None) -> None:
    """Limpa data/dl/sweep antes de um launch-train fresco."""
    root = resolve_repo_path(artifact_root, repo_root=repo_root)
    if root.is_dir():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def run_launch_train_tf_pipeline(
    *,
    only_tf: set[str] | None = None,
    skip_promote: bool = False,
    dry_run: bool = False,
) -> int:
    """Executa sweep e promove vencedor elegivel; 0 ok, !=0 falha fail-closed."""
    log = setup_logger("TF_SWEEP")
    settings = load_settings_json()
    knobs = load_tf_sweep_knobs(settings)
    if not bool(knobs.get("enabled", True)) or not bool(knobs.get("run_in_launch_train", True)):
        log.info("[TF_SWEEP] desligado; fallback train.py single-TF")
        if dry_run:
            return 0
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "train.py")],
            cwd=str(REPO_ROOT),
            check=False,
        )
        return int(proc.returncode)
    clear_sweep_artifacts(str(knobs["artifact_root"]), repo_root=REPO_ROOT)
    log.info(
        "[TF_SWEEP] iniciando sweep multi-TF (retries=%s; infra=%s)",
        knobs.get("train_deploy_retries"),
        "off" if knobs.get("disable_infra_during_sweep", True) else "on",
    )
    board = run_tf_sweep(knobs=knobs, only_tf=only_tf, dry_run=dry_run)
    eligible = [r for r in board if r.get("eligible")]
    log.info("[TF_SWEEP] candidatos=%s elegiveis=%s", len(board), len(eligible))
    for row in board:
        log.info(
            "[TF_SWEEP] tf=%s settle=%.4f n=%s label_acc=%.4f edge_vs_be=%.4f deploy=%s eligible=%s",
            row.get("tf"),
            float(row.get("rank_wr") or row.get("settle_wr") or 0.0),
            int(row.get("settle_n") or 0),
            float(row.get("val_accuracy") or 0.0),
            float(row.get("edge_vs_be") or 0.0),
            row.get("deploy_ok"),
            row.get("eligible"),
        )
    if dry_run:
        log.info("[TF_SWEEP] dry-run: sem promote")
        return 0
    winner = pick_tf_winner(board)
    if winner is None:
        log.error(
            "[TF_SWEEP] nenhum TF elegivel (settle_wr>=be+%.3f e settle_n>=%s); aborta launch-train",
            float(knobs["min_edge_vs_breakeven"]),
            int(knobs.get("min_settle_n", 16)),
        )
        return 1
    if skip_promote or not bool(knobs.get("auto_promote", True)):
        log.info(
            "[TF_SWEEP] winner=%s sem auto_promote; rode promote_tf_winner.py",
            winner.get("tf"),
        )
        return 0
    return int(promote_main([]))


def main(argv: list[str] | None = None) -> int:
    """CLI chamada pelo launch-train.bat."""
    parser = argparse.ArgumentParser(description="Sweep+promote para launch-train")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-promote", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    only = {t.strip().upper() for t in args.only} if args.only else None
    return run_launch_train_tf_pipeline(
        only_tf=only,
        skip_promote=bool(args.skip_promote),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
