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
    macro_confluence: str = "",
    fx_reference_line: str = "",
    macro_sentiment: str = "",
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
        macro_confluence=macro_confluence,
        fx_reference_line=fx_reference_line,
        macro_sentiment=macro_sentiment,
    )
    rows: list[tuple[str, str]] = [
        ("ATIVO", f"{symbol}"),
        ("SNIPER_INPUT", sniper_line),
        ("MACRO_CONFLUENCIA", (macro_confluence or "-").strip() or "-"),
        ("CONTEXTO_FX_REF", (fx_reference_line or "-").strip() or "-"),
        ("MACRO_SENTIMENT", (macro_sentiment or "-").strip() or "-"),
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
    macro_confluence: str = "",
    fx_reference_line: str = "",
    macro_sentiment: str = "",
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
    macro_block = f"{macro_confluence}\n" if (macro_confluence or "").strip() else ""
    fx_line = (fx_reference_line or "").strip()
    fx_block = ""
    if fx_line and "CONTEXTO_FX_REF" not in (macro_confluence or ""):
        fx_block = f"{fx_line}\n"
    sentiment_seg = f"sentiment={macro_sentiment} | " if (macro_sentiment or "").strip() else ""
    return (
        f"SNIPER: {core}\n"
        f"{matrix_block}"
        f"{macro_block}"
        f"{fx_block}"
        f"DATA: {symbol} | PAYOUT: {payout_estimate} (min: {min_payout_accept}) | DUR: {duration}{duration_unit}\n"
        f"REGIME: {(regime_line or '').strip()} | {(session_line or '').strip()} | {(micro_line or '').strip()}\n"
        f"{maps}"
        f"ALIGN: {mtf_align}\n"
        f"CANDLES: {candles_txt}\n"
        f"METRICS: {metrics_txt}\n"
        f"{sentiment_seg}"
        f"{f'CLUSTERS REALTIME: {cluster_status}\n' if cluster_status else ''}"
        f"{ib_block}{pa_block}"
        f"PERF: WR: {f'{float(wr_rolling):.1%}' if wr_rolling is not None else 'n/a'} ({int(wr_samples)})\n"
        "=== REGRAS DE TRADING ===\n"
        "- MACRO: Risk-On (US+EU em RISE) favorece EURUSD CALL e clusters CALL; Risk-Off (FALL) favorece PUT.\n"
        "- DIVERGENCIA: Se US e EU divergirem, reduza conviccao; EURUSD e clusters somente CALL ou PUT.\n"
        "- FX_REF: USDJPY, AUDUSD, NZDUSD usam RISE/FALL como contexto macro (sem ordens nesses pares).\n"
        "- ESTRUTURA: Priorize D1 e H4; M5 e M1 apenas timing.\n"
        "- HURST: Acima de 0.55 momentum; abaixo de 0.45 reversao via Z-Score extremo.\n"
        "- ENTROPIA: extreme_sigma em M1/M5 -> Probabilidade max 0.70; cluster afetado somente CALL ou PUT.\n"
        "- YIELDS: Treasuries/Bunds altos drenam acoes (Risk-Off); mencione apenas se MACRO indicar queda conjunta.\n"
        "=== SINTESE FINAL ===\n"
        "Responda OBRIGATORIAMENTE no formato: EURUSD: [DIR] | US_CLUSTER: [DIR] | EU_CLUSTER: [DIR] | Probabilidade: [0.XX]."
    )
