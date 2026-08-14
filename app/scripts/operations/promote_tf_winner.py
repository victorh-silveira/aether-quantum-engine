"""Promove o vencedor do leaderboard de horizonte para settings + data/dl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from aether_paths import REPO_ROOT
from src.application.services.deep_learning.tf_sweep_config import load_tf_sweep_knobs, resolve_repo_path
from src.application.services.deep_learning.tf_sweep_promote import (
    promote_winner_from_leaderboard,
    write_json,
)
from src.application.services.deep_learning.tf_sweep_score import enrich_leaderboard_row, pick_tf_winner
from src.domain.config_knobs import load_settings_json
from src.presentation.terminal.logger import setup_logger


def load_leaderboard(path: Path, knobs: dict) -> list[dict]:
    """Le leaderboard JSON e re-enriquece linhas."""
    if not path.is_file():
        raise FileNotFoundError(f"leaderboard ausente: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("rows"), list):
        raise ValueError("leaderboard invalido")
    return [enrich_leaderboard_row(row, knobs=knobs) for row in raw["rows"] if isinstance(row, dict)]


def main(argv: list[str] | None = None) -> int:
    """CLI de promocao fail-closed do TF vencedor."""
    parser = argparse.ArgumentParser(description="Promove vencedor do sweep de horizonte para SSOT")
    parser.add_argument("--dry-run", action="store_true", help="Nao grava settings nem copia ckpt")
    parser.add_argument(
        "--leaderboard",
        default=None,
        help="Path do leaderboard (default: deep_learning.horizon_sweep.leaderboard_path)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Override do simbolo promovido (default: coluna symbol do winner)",
    )
    args = parser.parse_args(argv)
    log = setup_logger("TF_SWEEP")
    settings = load_settings_json()
    knobs = load_tf_sweep_knobs(settings)
    board_path = Path(args.leaderboard) if args.leaderboard else resolve_repo_path(str(knobs["leaderboard_path"]))
    rows = load_leaderboard(board_path, knobs)
    preview = pick_tf_winner(rows)
    if preview is None:
        log.error(
            "[TF_SWEEP] nenhum TF elegivel (settle_wr>=be+%.3f); SSOT intacto",
            float(knobs["min_edge_vs_breakeven"]),
        )
        return 1
    log.info(
        "[TF_SWEEP] winner=%s/%s acc=%.4f edge_vs_be=%.4f score=%s",
        preview.get("symbol"),
        preview.get("tf"),
        float(preview.get("val_accuracy") or 0.0),
        float(preview.get("edge_vs_be") or 0.0),
        preview.get("score"),
    )
    if args.dry_run:
        log.info("[TF_SWEEP] dry-run: sem gravar settings/ckpt")
        return 0
    winner, patched, copied = promote_winner_from_leaderboard(
        rows,
        settings,
        artifact_root=str(knobs["artifact_root"]),
        symbol=str(args.symbol) if args.symbol else None,
        copy_artifacts=True,
    )
    if winner is None:
        log.error("[TF_SWEEP] promocao abortada: sem elegivel")
        return 1
    settings_path = REPO_ROOT / "config" / "settings.json"
    backup = settings_path.with_suffix(".json.pre_tf_promote.bak")
    backup.write_text(settings_path.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(settings_path, patched)
    log.info(
        "[TF_SWEEP] promovido symbol=%s tf=%s settings=%s backup=%s ckpts=%s",
        patched.get("anchor") or winner.get("symbol"),
        winner.get("tf"),
        settings_path,
        backup,
        [str(p) for p in copied],
    )
    log.info("[TF_SWEEP] rode make docker-rebuild e sync MinIO/Triton apos promocao")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
