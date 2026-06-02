"""Selecao Medallion de indices dentro do cluster ativo via Z-Score StatArb."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.cluster_statarb_score import alignment_score, wr_blend_score
from src.application.services.llm.llm_macro_confluence_guards import _statarb_misaligned
from src.application.services.llm.macro_config import resolve_macro_config
from src.domain.models.trade import TradeDirection


def resolve_statarb_cluster_config_for_tag(
    corr: dict[str, Any] | None,
    macro: dict[str, Any] | None,
    macro_tag: str,
) -> dict[str, Any]:
    """Mescla config StatArb com piso de |Z| especifico da tag macro ativa."""
    statarb_cfg = resolve_statarb_cluster_config(corr, macro)
    macro_dict = macro if isinstance(macro, dict) else {}
    by_tag = macro_dict.get("statarb_min_abs_z_by_tag")
    tag = str(macro_tag or "")
    if isinstance(by_tag, dict) and tag and tag in by_tag:
        statarb_cfg = dict(statarb_cfg)
        statarb_cfg["min_abs_z"] = max(0.0, float(by_tag[tag]))
    return statarb_cfg


def resolve_statarb_cluster_config(corr: dict[str, Any] | None, macro: dict[str, Any] | None) -> dict[str, Any]:
    """Mescla flags de selecao por indice em correlation e strategy.macro."""
    c = corr if isinstance(corr, dict) else {}
    m = resolve_macro_config(macro if isinstance(macro, dict) else None)
    return {
        "enabled": True,
        "execute_all": False,
        "best_symbol_only": True,
        "max_per_cluster": 1,
        "min_abs_z": max(0.0, float(c.get("statarb_index_min_abs_z", 0.0))),
        "wr_weight": max(0.0, float(c.get("statarb_wr_weight", 0.35))),
        "require_z_align": bool(c.get("statarb_require_z_align", False)),
        "z_align_soft_fallback": bool(c.get("statarb_z_align_soft_fallback", False)),
        "soft_min_abs_ratio": max(0.1, float(c.get("statarb_soft_min_abs_ratio", 0.45))),
        "weak_leader_on_no_align": False,
        "z_threshold": float(m["statarb_z_threshold"]),
    }


def _alignment_score(z: float, direction: TradeDirection, hmm_state: int) -> float:
    """Compat: delega pontuacao de alinhamento Z para cluster_statarb_score."""
    return alignment_score(z, direction, hmm_state)


def _wr_blend_score(sym: str, wr_scores: dict[str, float] | None, weight: float) -> float:
    """Compat: delega blend de win-rate rolling para cluster_statarb_score."""
    return wr_blend_score(sym, wr_scores, weight)


def statarb_execute_min_abs_z(index_note: str, statarb_cfg: dict[str, Any]) -> float:
    """Piso de |Z| no gate de execute; sempre exige confirmacao StatArb estrita na entrada."""
    _ = index_note
    return float(statarb_cfg.get("min_abs_z", 0.0))


def symbol_z_supports_direction(
    z: float,
    direction: TradeDirection,
    *,
    hmm_state: int = 0,
    z_threshold: float = 2.5,
    min_abs_z: float = 0.0,
) -> bool:
    """True quando o Z do indice confirma a direcao (MR ou tendencia HMM)."""
    zf = float(z)
    floor = max(0.0, float(min_abs_z))
    if abs(zf) < floor:
        return False
    if int(hmm_state) == 1:
        if direction == TradeDirection.CALL:
            return zf >= floor
        if direction == TradeDirection.PUT:
            return zf <= -floor
        return False
    if _statarb_misaligned(direction, zf, float(z_threshold), int(hmm_state)):
        return False
    return _alignment_score(zf, direction, int(hmm_state)) > 0.0


def _statarb_leader_pick(
    leader_rows: list[tuple[str, float, float]],
    *,
    wr_scores: dict[str, float] | None,
    best_symbol_only: bool,
) -> tuple[set[str], str]:
    """Monta conjunto e nota a partir das linhas lideres StatArb."""
    picked = {row[0] for row in leader_rows}
    leader = leader_rows[0]
    wr_part = ""
    if wr_scores and leader[0] in wr_scores:
        wr_part = f" wr={wr_scores[leader[0]]:.2f}"
    tag = "STATARB_BEST" if best_symbol_only else "STATARB_INDEX"
    note = f"{tag} leader={leader[0]} z={leader[1]:.2f} score={leader[2]:.2f}{wr_part} n={len(picked)}"
    return picked, note


def select_cluster_symbols_by_statarb(
    candidates: list[str],
    direction: TradeDirection,
    statarb_spreads: dict[str, float] | None,
    *,
    hmm_state: int = 0,
    cfg: dict[str, Any] | None = None,
    wr_scores: dict[str, float] | None = None,
) -> tuple[set[str], str]:
    """Retorna subconjunto de indices com melhor alinhamento StatArb ao cluster."""
    base_cfg = cfg if isinstance(cfg, dict) else {}
    if not base_cfg.get("enabled", True):
        return set(candidates), "CLUSTER_ALL_SYMBOLS"

    spreads = statarb_spreads or {}
    if not candidates:
        return set(), "STATARB_INDEX_EMPTY"

    wr_weight = float(base_cfg.get("wr_weight", 0.0))
    min_abs = float(base_cfg.get("min_abs_z", 0.0))
    require_align = bool(base_cfg.get("require_z_align", False))
    z_threshold = float(base_cfg.get("z_threshold", 2.5))
    soft_fallback = bool(base_cfg.get("z_align_soft_fallback", False))
    soft_ratio = float(base_cfg.get("soft_min_abs_ratio", 0.45))

    def run_selection(floor_z: float) -> list[tuple[str, float, float]]:
        """Filtra e pontua os candidatos de acordo com o piso de Z-Score especificado."""
        res = []
        for sym in candidates:
            z = spreads.get(sym)
            if z is None:
                continue
            zf = float(z)
            if require_align and not symbol_z_supports_direction(
                zf,
                direction,
                hmm_state=hmm_state,
                z_threshold=z_threshold,
                min_abs_z=floor_z,
            ):
                continue
            align = _alignment_score(zf, direction, hmm_state)
            composite = align + _wr_blend_score(sym, wr_scores, wr_weight)
            res.append((sym, zf, composite))
        return res

    scored = run_selection(min_abs)
    if not scored and require_align and soft_fallback:
        scored = run_selection(min_abs * soft_ratio)

    if not scored:
        if require_align:
            return set(), "STATARB_NO_Z_ALIGN"
        return set(candidates), "STATARB_INDEX_NO_Z_FALLBACK"

    ranked = sorted(scored, key=lambda row: (row[2], abs(row[1])), reverse=True)
    filtered = [row for row in ranked if row[2] > 0.0 or abs(row[1]) >= min_abs]

    max_n = 1
    if not filtered:
        return set(), "STATARB_NO_Z_ALIGN"

    return _statarb_leader_pick(
        filtered[:max_n],
        wr_scores=wr_scores,
        best_symbol_only=True,
    )
