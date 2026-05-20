"""Formatacao de contexto e prompt para decisao LLM Profunda."""

from __future__ import annotations

from src.application.services.llm.sniper_payload import (
    coerce_sniper_tokens,
    format_sniper_prompt_line,
)
from src.application.services.llm.strategy_payload_config import StrategyPayloadConfig


def _truncate_audit_line(text: str, max_chars: int) -> str:
    """Limita texto longo para linha de log sem quebrar encoding."""
    clean = " ".join((text or "").replace("\n", " ").split()).strip()
    if not clean:
        return "-"
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def format_metrics_line(
    hurst: float | None,
    zscore: float | None,
    entropy: float | None,
) -> str:
    """Formata METRICS com valores numericos do gatilho."""
    h_txt = f"{float(hurst):.2f}" if hurst is not None else "-"
    z_txt = f"{float(zscore):+.2f}" if zscore is not None else "-"
    e_txt = f"{float(entropy):.2f}" if entropy is not None else "-"
    return f"H={h_txt}, Z={z_txt}, E={e_txt}"


def compact_m1_candles_csv(
    m3_tail_closes: list[float], ohlc_rows: list[tuple[float, float, float, float]] | None = None
) -> str:
    """Replica CANDLES_M1_compact do prompt. Se houver OHLC, envia (O/H/L/C)."""
    if ohlc_rows:
        return " | ".join(f"({r[0]:.5f}/{r[1]:.5f}/{r[2]:.5f}/{r[3]:.5f})" for r in ohlc_rows[-6:])

    tail = [float(x) for x in (m3_tail_closes[-6:] if len(m3_tail_closes) >= 6 else m3_tail_closes)]
    return ",".join(f"{x:.5f}" for x in tail) if tail else "-"


def _maps_line(
    lm: str,
    macro_desc: str,
    ls: str,
    structure_desc: str,
    lw: str,
    swing_desc: str,
    lt: str,
    trigger_desc: str,
    l5: str,
    micro_swing_desc: str,
    l1: str,
    micro_trigger_desc: str,
) -> str:
    """Monta linha MAPS com rotulos dinamicos alinhados aos timeframes reais."""
    return (
        f"MAPS: {lm}:{macro_desc} | {ls}:{structure_desc} | {lw}:{swing_desc} | "
        f"{lt}:{trigger_desc} | {l5}:{micro_swing_desc} | {l1}:{micro_trigger_desc}\n"
    )


def iter_llm_prompt_audit_sections(
    symbol: str,
    macro_desc: str,
    structure_desc: str,
    swing_desc: str,
    trigger_desc: str,
    sniper_tokens: dict[str, str] | None,
    mtf_align: str,
    regime_line: str,
    session_line: str,
    micro_line: str,
    trigger_tail_closes: list[float],
    payout_estimate: float,
    min_payout_accept: float,
    duration: int | str,
    duration_unit: str,
    wr_rolling: float | None = None,
    wr_samples: int = 0,
    *,
    trigger_ohlc: list[tuple[float, float, float, float]] | None = None,
    max_chars: int = 900,
    strategy_payload: StrategyPayloadConfig | None = None,
    institutional_pa_bundle: str = "",
    indicator_bundle_line: str = "",
    micro_swing_desc: str = "",
    micro_trigger_desc: str = "",
    mtf_matrix: str = "",
    tf_labels: tuple[str, ...] = ("D1", "H4", "H1", "M15", "M5", "M1"),
    metrics_h: float | None = None,
    metrics_z: float | None = None,
    metrics_e: float | None = None,
) -> list[tuple[str, str]]:
    """Lista etiquetas espelhando o prompt sniper e contexto resumido."""
    lm, ls, lw, lt, l5, l1 = tf_labels if len(tf_labels) >= 6 else (*tf_labels, "M5", "M1")[:6]
    sniper_line = build_sniper_trading_prompt(
        symbol,
        macro_desc,
        structure_desc,
        swing_desc,
        trigger_desc,
        sniper_tokens,
        mtf_align,
        regime_line,
        session_line,
        micro_line,
        trigger_tail_closes,
        payout_estimate,
        min_payout_accept,
        duration,
        duration_unit,
        trigger_ohlc=trigger_ohlc,
        strategy_payload=strategy_payload,
        institutional_pa_bundle=institutional_pa_bundle,
        indicator_bundle_line=indicator_bundle_line,
        wr_rolling=wr_rolling,
        wr_samples=wr_samples,
        micro_swing_desc=micro_swing_desc,
        micro_trigger_desc=micro_trigger_desc,
        mtf_matrix=mtf_matrix,
        tf_labels=tf_labels,
        metrics_h=metrics_h,
        metrics_z=metrics_z,
        metrics_e=metrics_e,
    )
    rows: list[tuple[str, str]] = [
        ("ATIVO", f"{symbol}"),
        ("SNIPER_INPUT", sniper_line),
        ("MTF_MATRIX", (mtf_matrix or "-").strip() or "-"),
        ("PA_INSTITUCIONAL", (institutional_pa_bundle or "-").strip() or "-"),
        ("REGIME", regime_line),
        ("SESSAO", session_line),
        ("MICRO", micro_line),
        ("MAPA_TF_CASCADE", f"macro: {macro_desc} | estrutura: {structure_desc} | swing: {swing_desc}"),
        ("GATILHO_TF", trigger_desc),
        ("ALINHAMENTO", mtf_align),
        ("PAYOUT_MIN", f"PAYOUT: {payout_estimate} | MIN_PAYOUT: {min_payout_accept}"),
        ("CONTRATO", f"{duration}{duration_unit}"),
        ("INDICADORES_MULTITF_LOG", (indicator_bundle_line or "-").strip() or "-"),
    ]
    return [(tag, _truncate_audit_line(body, max_chars)) for tag, body in rows]


def build_sniper_trading_prompt(
    symbol: str,
    macro_desc: str,
    structure_desc: str,
    swing_desc: str,
    trigger_desc: str,
    sniper_tokens: dict[str, str] | None,
    mtf_align: str,
    regime_line: str,
    session_line: str,
    micro_line: str,
    trigger_tail_closes: list[float],
    payout_estimate: float,
    min_payout_accept: float,
    duration: int | str,
    duration_unit: str,
    *,
    trigger_ohlc: list[tuple[float, float, float, float]] | None = None,
    strategy_payload: StrategyPayloadConfig | None = None,
    institutional_pa_bundle: str = "",
    indicator_bundle_line: str = "",
    wr_rolling: float | None = None,
    wr_samples: int = 0,
    cluster_status: str = "",
    micro_swing_desc: str = "",
    micro_trigger_desc: str = "",
    mtf_matrix: str = "",
    tf_labels: tuple[str, ...] = ("D1", "H4", "H1", "M15", "M5", "M1"),
    metrics_h: float | None = None,
    metrics_z: float | None = None,
    metrics_e: float | None = None,
) -> str:
    """Monta prompt usuario completo (Sempre Profundo)."""
    labels = tf_labels if len(tf_labels) >= 6 else ("D1", "H4", "H1", "M15", "M5", "M1")
    lm, ls, lw, lt, l5, l1 = labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]
    ms_desc = micro_swing_desc or f"{l5} indisponivel"
    mt_desc = micro_trigger_desc or f"{l1} indisponivel"
    core = format_sniper_prompt_line(
        symbol,
        macro_desc,
        structure_desc,
        swing_desc,
        trigger_desc,
        coerce_sniper_tokens(sniper_tokens),
        strategy_payload,
        mtf_alignment_line=mtf_align,
        micro_swing_desc=ms_desc,
        micro_trigger_desc=mt_desc,
    )
    matrix_block = f"{mtf_matrix}\n" if (mtf_matrix or "").strip() else ""
    metrics_txt = format_metrics_line(metrics_h, metrics_z, metrics_e)
    candles_txt = compact_m1_candles_csv(trigger_tail_closes, ohlc_rows=trigger_ohlc)
    ib = (indicator_bundle_line or "").strip()
    pa = (institutional_pa_bundle or "").strip()
    ib_block = f"INDICADORES: {ib}\n" if ib else ""
    pa_block = f"CONFLUENCIA: {pa}\n" if pa else ""
    maps = _maps_line(lm, macro_desc, ls, structure_desc, lw, swing_desc, lt, trigger_desc, l5, ms_desc, l1, mt_desc)
    return (
        f"SNIPER: {core}\n"
        f"{matrix_block}"
        f"DATA: {symbol} | PAYOUT: {payout_estimate} (min: {min_payout_accept}) | DUR: {duration}{duration_unit}\n"
        f"REGIME: {(regime_line or '').strip()} | {(session_line or '').strip()} | {(micro_line or '').strip()}\n"
        f"{maps}"
        f"ALIGN: {mtf_align}\n"
        f"CANDLES: {candles_txt}\n"
        f"METRICS: {metrics_txt}\n"
        f"{f'CLUSTERS REALTIME: {cluster_status}\n' if cluster_status else ''}"
        f"{ib_block}{pa_block}"
        f"PERF: WR: {f'{float(wr_rolling):.1%}' if wr_rolling is not None else 'n/a'} ({int(wr_samples)})\n"
        "=== REGRAS DE TRADING ===\n"
        "- ESTRUTURA: Priorize alinhamento D1 e H4; M5 e M1 apenas para timing de entrada.\n"
        "- HURST: Acima de 0.55 siga momentum; abaixo de 0.45 priorize reversao via Z-Score extremo.\n"
        "- ENTROPIA: Se M1 ou M5 estiverem em extreme_sigma, reduza Probabilidade para no maximo 0.70 ou WAIT no cluster afetado.\n"
        "- TIMING: Evite entrar na expansao continua sem exaustao; exija desaceleracao ou vela contraria em reversao.\n"
        "- CLUSTERS: US_CLUSTER e EU_CLUSTER sao independentes do EURUSD quando os dados de cluster indicarem divergencia.\n"
        "=== SINTESE FINAL ===\n"
        "Responda OBRIGATORIAMENTE no formato: EURUSD: [DIR] | US_CLUSTER: [DIR] | EU_CLUSTER: [DIR] | Probabilidade: [0.XX]."
    )


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
