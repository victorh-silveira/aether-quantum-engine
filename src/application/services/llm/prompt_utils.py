"""Formatacao de contexto e prompt para decisao LLM Profunda."""

from __future__ import annotations

import re

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


def extract_prompt_indicator_tokens(m3_desc: str, entropy_val: float | None) -> tuple[str, str, str]:
    """Extrai Hurst e Z-Score alinhados ao texto do prompt enviado a LLM."""
    hurst_value = "-"
    zscore_val = "-"
    m = re.search(r"Hurst(?:=|:)\s*([0-9]+(?:[.,][0-9]+)?)", m3_desc, flags=re.IGNORECASE)
    if m:
        hurst_value = m.group(1).replace(",", ".")
    d = re.search(r"Z-Score(?:=|:)\s*([+-]?[0-9]+(?:[.,][0-9]+)?)", m3_desc, flags=re.IGNORECASE)
    if d:
        zscore_val = d.group(1).replace(",", ".")
    ent_txt = f"{float(entropy_val):.2f}" if entropy_val is not None else "-"
    return hurst_value, zscore_val, ent_txt


def compact_m1_candles_csv(
    m3_tail_closes: list[float], ohlc_rows: list[tuple[float, float, float, float]] | None = None
) -> str:
    """Replica CANDLES_M1_compact do prompt. Se houver OHLC, envia (O/H/L/C)."""
    if ohlc_rows:
        return " | ".join(f"({r[0]:.5f}/{r[1]:.5f}/{r[2]:.5f}/{r[3]:.5f})" for r in ohlc_rows[-6:])

    tail = [float(x) for x in (m3_tail_closes[-6:] if len(m3_tail_closes) >= 6 else m3_tail_closes)]
    return ",".join(f"{x:.5f}" for x in tail) if tail else "-"


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
    atr_m5_pct: float | None,
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
) -> list[tuple[str, str]]:
    """Lista etiquetas espelhando o prompt sniper e contexto resumido."""
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
        atr_m5_pct,
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
    )
    rows: list[tuple[str, str]] = [
        ("ATIVO", f"{symbol}"),
        ("SNIPER_INPUT", sniper_line),
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
    atr_m5_pct: float | None,
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
) -> str:
    """Monta prompt usuario completo (Sempre Profundo)."""
    core = format_sniper_prompt_line(
        symbol,
        macro_desc,
        structure_desc,
        swing_desc,
        trigger_desc,
        coerce_sniper_tokens(sniper_tokens),
        strategy_payload,
        mtf_alignment_line=mtf_align,
    )
    hurst_val, zscore_val, ent_txt = extract_prompt_indicator_tokens(trigger_desc, atr_m5_pct)
    candles_txt = compact_m1_candles_csv(trigger_tail_closes, ohlc_rows=trigger_ohlc)
    ib = (indicator_bundle_line or "").strip()
    pa = (institutional_pa_bundle or "").strip()
    ib_block = f"INDICADORES: {ib}\n" if ib else ""
    pa_block = f"CONFLUENCIA: {pa}\n" if pa else ""
    return (
        f"SNIPER: {core}\n"
        f"DATA: {symbol} | PAYOUT: {payout_estimate} (min: {min_payout_accept}) | DUR: {duration}{duration_unit}\n"
        f"REGIME: {(regime_line or '').strip()} | {(session_line or '').strip()} | {(micro_line or '').strip()}\n"
        f"MAPS: M30:{macro_desc} | M15:{structure_desc} | M5:{swing_desc} | M1:{trigger_desc}\n"
        f"ALIGN: {mtf_align}\n"
        f"CANDLES: {candles_txt}\n"
        f"METRICS: H={hurst_val}, Z={zscore_val}, E={ent_txt}\n"
        f"{ib_block}{pa_block}"
        f"PERF: WR: {f'{float(wr_rolling):.1%}' if wr_rolling is not None else 'n/a'} ({int(wr_samples)})\n"
        "=== REGRAS DE TRADING ===\n"
        "- Em regime de reversão à média, evite operar contra a tendência se a velocidade for forte e não houver sinal de exaustão.\n"
        "- Só valide inversões se houver desaceleração ou vela contrária recente.\n"
        "- IMPORTANTE: Pondere as correlações! Se o Dólar está forte (EURUSD PUT), isso pode punir o US_CLUSTER (PUT) mas favorecer as exportações europeias no EU_CLUSTER (CALL). Avalie cada cenário separadamente.\n"
        "=== SÍNTESE FINAL ===\n"
        "Responda OBRIGATORIAMENTE no formato: EURUSD: [DIR] | US_CLUSTER: [DIR] | EU_CLUSTER: [DIR] | Probabilidade: [0.XX]."
    )


def build_institutional_pa_bundle(
    *,
    regime_label: str,
    atr_m5_pct: float | None,
    indicators_numeric_line: str,
    cf_dual: str,
    line_macro_structure: str,
    line_swing_trigger: str,
    compact: bool = False,
) -> str:
    """Monta pacote PA para o prompt Profundo (ignora compact)."""
    _ = compact
    atr_seg = f"atr_swing_pct={float(atr_m5_pct):.4f}" if atr_m5_pct is not None else "atr_swing_pct=-"
    parts: list[str] = [
        f"LLM_DADOS_NUM={indicators_numeric_line}",
        f"reg={regime_label}",
        atr_seg,
        cf_dual,
        f"CONFL_MACRO_ESTRUTURA={line_macro_structure}",
        f"CONFL_SWING_GATILHO={line_swing_trigger}",
    ]
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
    h_val, z_val, e_txt = extract_prompt_indicator_tokens(m3_desc, entropy_val)
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
        f"INDICADORES: Hurst={h_val}, Z-Score={z_val}, Entropy={e_txt}\n"
        f"PAYOUT: {payout_estimate} | MIN_PAYOUT: {min_payout_accept}\n"
        f"CONTRATO: {duration}{duration_unit}\n"
        "MEDALLION V15: Decida com base em arbitragem estatística. "
        "Responda EXCLUSIVAMENTE: CALL ou PUT."
    )
