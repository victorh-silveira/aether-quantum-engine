"""Pipeline launch-train: sweep de horizonte N + promote (fail-closed)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from aether_paths import REPO_ROOT
from scripts.operations.promote_tf_winner import main as promote_main
from scripts.operations.sweep_train_timeframes import run_tf_sweep
from src.application.services.deep_learning.horizon_sweep import (
    build_horizon_candidates,
    load_horizon_sweep_knobs,
)
from src.application.services.deep_learning.tf_sweep_config import resolve_repo_path
from src.application.services.deep_learning.tf_sweep_score import pick_tf_winner
from src.domain.config_knobs import load_settings_json
from src.presentation.terminal.logger import setup_logger


def clear_sweep_artifacts(artifact_root: str, *, repo_root=None) -> None:
    """Limpa data/dl/sweep antes de um launch-train fresco."""
    root = resolve_repo_path(artifact_root, repo_root=repo_root)
    if root.is_dir():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _sweep_active(knobs: dict[str, Any]) -> bool:
    return bool(knobs.get("enabled", True)) and bool(knobs.get("run_in_launch_train", True))


def _log_board(log, tag: str, board: list[dict[str, Any]], _knobs: dict[str, Any]) -> None:
    eligible = [r for r in board if r.get("eligible")]
    log.info("[%s] candidatos=%s elegiveis=%s", tag, len(board), len(eligible))
    for row in board:
        log.info(
            "[%s] symbol=%s tf=%s N=%s duration=%sm settle=%.4f n=%s "
            "label_acc=%.4f edge_vs_be=%.4f deploy=%s eligible=%s",
            tag,
            row.get("symbol"),
            row.get("tf"),
            int(row.get("label_horizon_bars") or 0),
            int(row.get("duration") or 0),
            float(row.get("rank_wr") or row.get("settle_wr") or 0.0),
            int(row.get("settle_n") or 0),
            float(row.get("val_accuracy") or 0.0),
            float(row.get("edge_vs_be") or 0.0),
            row.get("deploy_ok"),
            row.get("eligible"),
        )


def _finish_sweep(
    log,
    tag: str,
    board: list[dict[str, Any]],
    knobs: dict[str, Any],
    *,
    skip_promote: bool,
    dry_run: bool,
) -> int:
    _log_board(log, tag, board, knobs)
    if dry_run:
        log.info("[%s] dry-run: sem promote", tag)
        return 0
    winner = pick_tf_winner(board)
    if winner is None:
        log.error(
            "[%s] nenhum candidato elegivel (settle_wr>=be+%.3f e settle_n>=%s); aborta launch-train",
            tag,
            float(knobs["min_edge_vs_breakeven"]),
            int(knobs.get("min_settle_n", 16)),
        )
        return 1
    n_bars = int(winner.get("label_horizon_bars") or 0)
    duration = int(winner.get("duration") or 0)
    log.info(
        "[%s] winner N=%s duration=%sm tf=%s settle=%.4f edge_vs_be=%.4f (mais assertivo)",
        tag,
        n_bars,
        duration,
        winner.get("tf"),
        float(winner.get("rank_wr") or winner.get("settle_wr") or 0.0),
        float(winner.get("edge_vs_be") or 0.0),
    )
    if skip_promote or not bool(knobs.get("auto_promote", True)):
        log.info(
            "[%s] winner=%s/%s sem auto_promote; rode promote_tf_winner.py",
            tag,
            winner.get("symbol"),
            winner.get("tf"),
        )
        return 0
    return int(promote_main([]))


def run_launch_train_tf_pipeline(
    *,
    only_tf: set[str] | None = None,
    skip_promote: bool = False,
    dry_run: bool = False,
) -> int:
    """Executa sweep de horizonte N e promove vencedor; 0 ok, !=0 falha fail-closed."""
    log = setup_logger("HORIZON")
    settings = load_settings_json()
    h_knobs = load_horizon_sweep_knobs(settings)
    if _sweep_active(h_knobs):
        clear_sweep_artifacts(str(h_knobs["artifact_root"]), repo_root=REPO_ROOT)
        candidates = build_horizon_candidates(settings, n_bars=h_knobs["n_bars"])
        log.info("[HORIZON] iniciando sweep N=%s (R_10 M3)", h_knobs.get("n_bars"))
        board = run_tf_sweep(
            settings=settings,
            knobs=h_knobs,
            candidates=candidates,
            only_tf=only_tf,
            dry_run=dry_run,
        )
        return _finish_sweep(log, "HORIZON", board, h_knobs, skip_promote=skip_promote, dry_run=dry_run)
    log.info("[HORIZON] desligado; fallback train.py")
    if dry_run:
        return 0
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "train.py")],
        cwd=str(REPO_ROOT),
        check=False,
    )
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    """CLI chamada pelo launch-train.bat."""
    parser = argparse.ArgumentParser(description="Sweep de horizonte + promote para launch-train")
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
