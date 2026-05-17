"""Linhas numericas compactas de indicadores para logs INFO."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

import src.application.services.llm.indicators as ti
from src.application.services.llm import IndicatorConfig


def trend_token_from_label_word(word: str) -> str:
    """Converte rotulo de regime quant em uma letra para log compacto."""
    w = (word or "").lower().strip()
    if "trend" in w or "persist" in w:
        return "P"
    if "mean" in w or "rev" in w:
        return "M"
    if "random" in w or "walk" in w or "noise" in w or "choppy" in w:
        return "N"
    return "?"


def abbrev_mtf_alignment_tokens(alignment: str) -> str:
    """Resume ALINHAMENTO M30|M5|M1 como A alta, B baixa, ? indefinido."""
    letters: list[str] = []
    for seg in (alignment or "").split("|"):
        chunk = seg.strip()
        if ":" not in chunk:
            continue
        rhs = chunk.split(":", 1)[1].strip()
        tw = rhs.split()[0] if rhs else ""
        letters.append(trend_token_from_label_word(tw))
    if not letters:
        return "-"
    return "/".join(letters)


def extract_confluence_heuristic_tag(confluence_line: str) -> str:
    """Extrai token apos sinal_quant= na linha de confluencia."""
    m = re.search(r"sinal_quant=([^|]+)", confluence_line or "")
    return m.group(1).strip() if m else "-"


def dual_confluence_shrunk_tags(line_m30_m5: str, line_m5_m1: str) -> str:
    """Retorna string cf30_5=...|cf5_1=... a partir das linhas de confluencia."""
    a = _shrink_cf_tag(extract_confluence_heuristic_tag(line_m30_m5))
    b = _shrink_cf_tag(extract_confluence_heuristic_tag(line_m5_m1))
    return f"cf30_5={a}|cf5_1={b}"


def _numeric_tf_segment(label: str, closes: Sequence[float], cfg: IndicatorConfig) -> str:
    """Segmento compacto entropy/hurst/zscore para um timeframe."""
    c = np.asarray(list(closes), dtype=np.float64)
    need = ti.min_bars_for_indicators(cfg)
    if c.size < need:
        return f"{label}=na"

    ent = ti._shannon_entropy(c, cfg.entropy_bins, cfg.entropy_window)
    hst = ti._hurst_exponent(c, cfg.hurst_window)
    zsc = ti._z_score_last(c, cfg.zscore_window)

    es = f"{ent:.2f}" if ent is not None else "na"
    hs = f"{hst:.2f}" if hst is not None else "na"
    zs = f"{zsc:+.1f}" if zsc is not None else "na"
    return f"{label} e{es} h{hs} z{zs}"


def format_numeric_indicators_one_line(
    macro: Sequence[float],
    structure: Sequence[float],
    swing: Sequence[float],
    trigger: Sequence[float],
    cfg: IndicatorConfig,
    atr_swing_pct: float | None,
    mtf_align_line: str,
    confluence_line: str,
    *,
    tf_labels: tuple[str, str, str, str] | None = None,
) -> str:
    """Uma unica linha numerica para log: quatro TF, atr, MTF e heuristica."""
    labs = tf_labels if tf_labels is not None else cfg.tf_labels
    parts = (
        _numeric_tf_segment(labs[0], macro, cfg),
        _numeric_tf_segment(labs[1], structure, cfg),
        _numeric_tf_segment(labs[2], swing, cfg),
        _numeric_tf_segment(labs[3], trigger, cfg),
    )
    atr = f"{float(atr_swing_pct):.4f}" if atr_swing_pct is not None else "-"
    mt = abbrev_mtf_alignment_tokens(mtf_align_line)
    cf = extract_confluence_heuristic_tag(confluence_line)
    return f"{parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} | sigma_m5={atr} | mtf={mt} | cf={cf}"


def _shrink_cf_tag(tag: str) -> str:
    """Encurta token de confluencia para linha de log."""
    t = (tag or "").strip()
    if not t or t == "-":
        return "-"
    mp = {
        "FORTE_CONTINUIDADE_QUANT (High Conviction)": "fQuant",
        "ARBITRAGEM_REVERSAO_ESTATISTICA (Counter-Trend)": "arbRev",
        "RANDOM_WALK_SEM_EDGE (Noisy)": "rWalk",
        "HIGH_ENTROPY_NOISE (Avoid High Stakes)": "noiseHi",
        "DIVERGENCIA_ESTRUTURAL_DETECTADA (Risky)": "divRisk",
        "dados_insuficientes": "noDat",
    }
    return mp.get(t, t[:18])


def _numeric_tf_tight(tag: str, closes: Sequence[float], cfg: IndicatorConfig) -> str:
    """Hurst, Z-Score e Velocidade; sem rotulos longos."""
    c = np.asarray(list(closes), dtype=np.float64)
    need = ti.min_bars_for_indicators(cfg)
    if c.size < need:
        return f"{tag}:na"

    hst = ti._hurst_exponent(c, cfg.hurst_window)
    zsc = ti._z_score_last(c, cfg.zscore_window)
    vel, _ = ti._price_derivatives(c, cfg.velocity_window)

    hs = f"{hst:.2f}" if hst is not None else "na"
    zs = f"{zsc:+.1f}" if zsc is not None else "na"
    vs = f"{vel:+.5f}" if vel is not None else "na"

    return f"{tag}:{hs}/{zs}/{vs}"


def format_numeric_indicators_tight_line(
    macro: Sequence[float],
    structure: Sequence[float],
    swing: Sequence[float],
    trigger: Sequence[float],
    cfg: IndicatorConfig,
    atr_swing_pct: float | None,
    mtf_align_line: str,
    confluence_line: str,
    *,
    tf_tags: tuple[str, str, str, str] | None = None,
) -> str:
    """Versao minima para INFO: quatro TF, atr no swing, mtf e cf com separadores espacados."""
    tags = tf_tags if tf_tags is not None else cfg.tf_tags
    pieces: list[str] = []
    for tag, series in zip(tags, (macro, structure, swing, trigger), strict=True):
        pieces.append(_numeric_tf_tight(tag, series, cfg))
    if atr_swing_pct is not None:
        pieces.append(f"sigma={float(atr_swing_pct):.3f}")
    mt = abbrev_mtf_alignment_tokens(mtf_align_line)
    if mt != "-":
        pieces.append(f"mtf={mt}")
    cf_raw = extract_confluence_heuristic_tag(confluence_line)
    if cf_raw != "-":
        cf = _shrink_cf_tag(cf_raw)
        if cf not in ("-", "noDat"):
            pieces.append(f"cf={cf}")
    return " | ".join(pieces) if pieces else "ind insuficiente"
