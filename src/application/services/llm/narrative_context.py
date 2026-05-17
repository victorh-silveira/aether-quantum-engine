"""Contexto narrativo SMC a partir de quatro graus de fechamento (macro a gatilho) para o prompt."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from src.application.services.llm import (
    IndicatorConfig,
    min_bars_for_indicators,
)
from src.application.services.llm.indicators import (
    _hurst_exponent,
    _price_derivatives,
    _shannon_entropy,
    _z_score_last,
)
from src.application.services.llm.regime import classify_regime, sigma_pct_m5


def _arr(closes: list[float]) -> np.ndarray:
    """Converte lista de fechamentos em ndarray float64."""
    return np.asarray(list(closes), dtype=np.float64)


def _statistical_anomaly_hint(z: float | None) -> str:
    """Indica se há uma anomalia estatística baseada no Z-Score."""
    if z is None:
        return "anomalia n/d"
    if z > 2.0:
        return f"anomalia_positiva (Z={z:.2f}) - potencial exaustao"
    if z < -2.0:
        return f"anomalia_negativa (Z={z:.2f}) - potencial exaustao"
    return "distribuicao_normal"


def describe_m15_map(closes: list[float], cfg: IndicatorConfig, tf_label: str = "H1") -> str:
    """Narrativa de mapa quant: Hurst (Persistencia) e Z-Score (Reversao)."""
    c = _arr(closes)
    need = min_bars_for_indicators(cfg)
    if c.size < need:
        return f"{tf_label} mapa indisponivel (amostra_curta n={c.size})."
    hurst = _hurst_exponent(c, cfg.hurst_window)
    zscore = _z_score_last(c, cfg.zscore_window)
    hint = _statistical_anomaly_hint(zscore)
    h_txt = f"{hurst:.2f}" if hurst is not None else "n/d"
    return f"{tf_label} mapa: hurst={h_txt}; {hint}; close={float(c[-1]):.5f}"


def describe_m5_filter(closes: list[float], cfg: IndicatorConfig, tf_label: str = "M15") -> str:
    """Narrativa de filtro quant: Entropia (Ruido) e Velocidade do Preco."""
    c = _arr(closes)
    need = min_bars_for_indicators(cfg)
    if c.size < need:
        return f"{tf_label} filtro indisponivel (amostra_curta n={c.size})."
    entropy = _shannon_entropy(c, cfg.entropy_bins, cfg.entropy_window)
    vel, _ = _price_derivatives(c, cfg.velocity_window)
    e_txt = f"{entropy:.2f}" if entropy is not None else "n/d"
    v_txt = f"{vel:+.6f}" if vel is not None else "n/d"
    noise = "RUIDO_ALTO" if entropy is not None and entropy > 2.5 else "PADRAO_DETECTAVEL"
    return f"{tf_label} filtro: entropia={e_txt} ({noise}); velocidade={v_txt}; close={float(c[-1]):.5f}"


def describe_m3_trigger(closes: list[float], cfg: IndicatorConfig, tf_label: str = "M5") -> str:
    """Narrativa de gatilho quant: Aceleracao e Z-Score de Curto Prazo."""
    c = _arr(closes)
    need = min_bars_for_indicators(cfg)
    if c.size < need:
        return f"{tf_label} gatilho indisponivel (amostra_curta n={c.size})."
    _, accel = _price_derivatives(c, cfg.acceleration_window)
    zscore = _z_score_last(c, cfg.zscore_window)
    a_txt = f"{accel:+.6f}" if accel is not None else "n/d"
    z_txt = f"{zscore:+.2f}" if zscore is not None else "n/d"
    last = float(c[-1])
    prev = float(c[-2]) if c.size >= 2 else last
    body = "bullish" if last > prev else "bearish" if last < prev else "doji"
    return f"{tf_label} gatilho: aceleracao={a_txt}; zscore={z_txt}; vela={body}; close={last:.5f}."


def _alignment_label_quant(closes: np.ndarray, cfg: IndicatorConfig) -> str:
    """Rotulo de tendencia quant para o resumo MTF."""
    need = min_bars_for_indicators(cfg)
    if closes.size < need:
        return "indefinido"
    hurst = _hurst_exponent(closes, cfg.hurst_window)
    zscore = _z_score_last(closes, cfg.zscore_window)

    if hurst is not None and hurst > 0.52:
        return "Momentum Alpha (Bull)" if (zscore or 0) > 0 else "Momentum Alpha (Bear)"
    if hurst is not None and hurst < 0.48:
        return "Mean Reversion Alpha"
    return "random_walk"


def describe_mtf_alignment(
    macro_closes: list[float],
    structure_closes: list[float],
    swing_closes: list[float],
    trigger_closes: list[float],
    cfg: IndicatorConfig,
    lm: str,
    ls: str,
    lw: str,
    lt: str,
) -> str:
    """Resume alinhamento quant por timeframe."""
    ma = _arr(macro_closes)
    sa = _arr(structure_closes)
    wa = _arr(swing_closes)
    ta = _arr(trigger_closes)
    tm = _alignment_label_quant(ma, cfg)
    ts = _alignment_label_quant(sa, cfg)
    tw = _alignment_label_quant(wa, cfg)
    tt = _alignment_label_quant(ta, cfg)
    return f"{lm}: {tm} | {ls}: {ts} | {lw}: {tw} | {lt}: {tt}"


def describe_volatility_regime(structure_closes: list[float], swing_closes: list[float], cfg: IndicatorConfig) -> str:
    """Texto de regime quant com classe e Sigma proxy de volatilidade."""
    cls = classify_regime(structure_closes, swing_closes, cfg)
    ent = sigma_pct_m5(swing_closes, cfg)
    ent_txt = f"{ent:.2f}" if ent is not None else "n/d"
    return f"REGIME_quant={cls}; sigma_swing={ent_txt}"


def describe_session_context() -> str:
    """Indica horario UTC atual e janela de sessao para o prompt."""
    now = datetime.now(UTC)
    h = int(now.hour)
    if 0 <= h < 7:
        win = "asia"
    elif 7 <= h < 13:
        win = "europa"
    elif 13 <= h < 21:
        win = "ny"
    else:
        win = "pos_ny"
    return f"SESSAO_UTC={now.strftime('%H:%M')}; janela={win}"


def _micro_swing_tag(tail: np.ndarray) -> str:
    """Resume microestrutura dos ultimos fechamentos com rotulo de swing."""
    if tail.size < 4:
        return "n/d"
    xs = [float(tail[i]) for i in range(tail.size)]
    hi_i = int(np.argmax(tail))
    lo_i = int(np.argmin(tail))
    last = xs[-1]
    if hi_i == len(xs) - 1 and lo_i < hi_i:
        return "HH_recente"
    if lo_i == len(xs) - 1 and hi_i < lo_i:
        return "LL_recente"
    if last >= max(xs[0], xs[-2]):
        return "pressao_alta"
    if last <= min(xs[0], xs[-2]):
        return "pressao_baixa"
    return "misturado"


def describe_micro_structure(closes: list[float], cfg: IndicatorConfig, tf_label: str = "M1") -> str:
    """Formata ultimos fechamentos no gatilho com tag de swing local."""
    c = _arr(closes)
    need = max(6, min_bars_for_indicators(cfg))
    if c.size < need:
        return f"MICRO_{tf_label}=n/d"
    tail = c[-6:]
    seq = ",".join(f"{float(x):.5f}" for x in tail)
    return f"MICRO_{tf_label} ultimos6={seq}; swing={_micro_swing_tag(tail)}"
