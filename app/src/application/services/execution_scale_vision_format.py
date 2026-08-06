"""Formatadores de telemetria SCALE para IND/CLUSTER."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_scale_micro import micro_regime_token


def format_scale_audit_line(metrics: dict[str, Any] | None) -> str:
    """Linha SCALE para log IND/CLUSTER."""
    m = metrics if isinstance(metrics, dict) else {}
    adapted = 1 if bool(m.get("scale_adapted")) else 0
    micro = micro_regime_token(m.get("scale_micro_regime"))
    return (
        f"SCALE || MACRO={m.get('scale_macro_dir') or '-'} "
        f"MICRO={m.get('scale_micro_dir') or '-'} "
        f"MINI={m.get('scale_mini_dir') or '-'} "
        f"MILI={m.get('scale_mili_dir') or '-'} "
        f"mi_prev={m.get('scale_mini_prev_bar_dir') or '-'} "
        f"mi_cur={m.get('scale_mini_bar_dir') or '-'} "
        f"mc_prev={m.get('scale_micro_prev_bar_dir') or '-'} "
        f"mc_cur={m.get('scale_micro_bar_dir') or '-'} "
        f"tape={m.get('scale_tape_consensus') or '-'} "
        f"micro={micro} "
        f"agree={int(m.get('scale_agree_n') or 0)}/4 "
        f"discord={bool(m.get('scale_discordance'))} "
        f"adapted={adapted}"
    )


def format_scale_ind_token(metrics: dict[str, Any] | None) -> str:
    """Token condensado SCALE para linha IND."""
    m = metrics if isinstance(metrics, dict) else {}
    tcn = m.get("tcn_direction") or m.get("scale_micro_dir") or "-"
    tape = m.get("scale_tape_consensus") or "-"
    adapted = 1 if bool(m.get("scale_adapted")) else 0
    mi_p = m.get("scale_mini_prev_bar_dir") or "-"
    mi = m.get("scale_mini_bar_dir") or m.get("scale_mini_dir") or "-"
    mili = m.get("scale_mili_dir") or "-"
    micro = micro_regime_token(m.get("scale_micro_regime"))
    vc = m.get("scale_vote_call_n")
    vp = m.get("scale_vote_put_n")
    vote = f" votes=C{vc}/P{vp}" if vc is not None and vp is not None else ""
    return f"SCALE: tcn={tcn} tape={tape} adapted={adapted} micro={micro} mi_p={mi_p} mi={mi} mili={mili}{vote}"
