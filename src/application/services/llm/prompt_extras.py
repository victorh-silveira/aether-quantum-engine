"""Funcoes auxiliares de prompt legacy e pacote institucional."""

from __future__ import annotations

from src.application.services.llm.prompt_utils import compact_m1_candles_csv, format_metrics_line


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


def build_trading_prompt(
    symbol: str,
    m15_desc: str,
    m5_desc: str,
    m3_desc: str,
    mtf_align: str,
    regime_line: str,
    session_line: str,
    micro_line: str,
    entropy_val: float | None,
    m3_tail_closes: list[float],
    payout_estimate: float,
    min_payout_accept: float,
    duration: int | str,
    duration_unit: str,
) -> str:
    """Monta texto do prompt V15 legacy (Atualizado Medallion)."""
    metrics_txt = format_metrics_line(None, None, entropy_val)
    candles_txt = compact_m1_candles_csv(m3_tail_closes)
    return (
        f"ATIVO: {symbol}\n"
        f"{regime_line}\n"
        f"{session_line}\n"
        f"{micro_line}\n"
        f"ESTRUTURA: {m15_desc} | FILTRO: {m5_desc}\n"
        f"GATILHO_QUANT: {m3_desc}\n"
        f"ALINHAMENTO: {mtf_align}\n"
        f"CANDLES_compact: {candles_txt}\n"
        f"INDICADORES: {metrics_txt}\n"
        f"PAYOUT: {payout_estimate} | MIN_PAYOUT: {min_payout_accept}\n"
        f"CONTRATO: {duration}{duration_unit}\n"
        "MEDALLION V15: Decida com base em arbitragem estatística. "
        "Responda EXCLUSIVAMENTE: CALL ou PUT."
    )
