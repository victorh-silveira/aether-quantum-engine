"""Pacote institucional de contexto quant para o prompt Medallion."""

from __future__ import annotations


def build_institutional_pa_bundle(
    *,
    regime_label: str,
    entropy_swing: float | None,
    vol_range_pct: float | None,
    indicators_numeric_line: str,
    cf_dual: str,
    line_macro_structure: str,
    line_swing_trigger: str,
    ema_guard: str = "",
    compact: bool = False,
) -> str:
    """Monta pacote PA para o prompt Profundo (ignora compact)."""
    _ = compact
    ent_seg = f"entropy_swing={float(entropy_swing):.4f}" if entropy_swing is not None else "entropy_swing=-"
    vol_seg = f"vol_range_pct={float(vol_range_pct):.3f}" if vol_range_pct is not None else "vol_range_pct=-"
    parts: list[str] = [
        f"LLM_DADOS_NUM={indicators_numeric_line}",
        f"reg={regime_label}",
        ent_seg,
        vol_seg,
        cf_dual,
        f"CONFL_MACRO_ESTRUTURA={line_macro_structure}",
        f"CONFL_SWING_GATILHO={line_swing_trigger}",
    ]
    guard = (ema_guard or "").strip()
    if guard and guard != "-":
        parts.append(f"ema_guard={guard}")
    return " || ".join(parts)
