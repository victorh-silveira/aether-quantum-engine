"""Tokens simplicados (sem valores crus) para o prompt do modelo Sniper."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import src.application.services.llm.indicators as ti
from src.application.services.llm import IndicatorConfig
from src.application.services.llm.strategy_payload_config import DEFAULT_STRATEGY_PAYLOAD_CONFIG, StrategyPayloadConfig


def hurst_token(closes: Sequence[float], ic: IndicatorConfig) -> str:
    """Classifica a persistencia fractal do preco."""
    arr = np.asarray(list(closes), dtype=np.float64)
    h = ti._hurst_exponent(arr, ic.hurst_window)
    if h is None:
        return "na"
    if h > 0.55:
        return "persist"
    if h < 0.45:
        return "anti"
    return "random"


def zscore_token(closes: Sequence[float], ic: IndicatorConfig) -> str:
    """Classifica o desvio estatistico (reversao a media)."""
    arr = np.asarray(list(closes), dtype=np.float64)
    z = ti._z_score_last(arr, ic.zscore_window)
    if z is None:
        return "na"
    if z > 2.0:
        return "high"
    if z < -2.0:
        return "low"
    return "neutral"


def entropy_token(closes: Sequence[float], ic: IndicatorConfig) -> str:
    """Classifica o nivel de ruído/aleatoriedade."""
    arr = np.asarray(list(closes), dtype=np.float64)
    e = ti._shannon_entropy(arr, ic.entropy_bins, ic.entropy_window)
    if e is None:
        return "na"
    if e > 2.8:
        return "extreme"
    if e > 2.4:
        return "high"
    return "low"


def velocity_token(closes: Sequence[float], ic: IndicatorConfig) -> str:
    """Classifica a inercia (velocidade) do preco."""
    arr = np.asarray(list(closes), dtype=np.float64)
    v, _ = ti._price_derivatives(arr, ic.velocity_window)
    if v is None:
        return "na"
    return "pos" if v > 0 else "neg"


def coerce_sniper_tokens(raw: object) -> dict[str, str]:
    """Preenche defaults para hurst, zscore, entropy e velocity."""
    out = {"hurst": "na", "zscore": "na", "entropy": "na", "velocity": "na"}
    if isinstance(raw, dict):
        for k in out:
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                out[k] = v.strip()
    return out


def build_sniper_tokens(
    m1_closes: Sequence[float],
    ic: IndicatorConfig,
    sp: StrategyPayloadConfig | None = None,
) -> dict[str, str]:
    """Monta mapa hurst, zscore, entropy, velocity com rotulos vindos da estrategia JSON."""
    cfg = sp or DEFAULT_STRATEGY_PAYLOAD_CONFIG

    h_c = hurst_token(m1_closes, ic)
    z_c = zscore_token(m1_closes, ic)
    e_c = entropy_token(m1_closes, ic)
    v_c = velocity_token(m1_closes, ic)

    h_map = {"persist": cfg.hurst_persist, "anti": cfg.hurst_anti, "random": cfg.hurst_random, "na": cfg.hurst_na}
    z_map = {"high": cfg.zscore_high, "low": cfg.zscore_low, "neutral": cfg.zscore_neutral, "na": cfg.zscore_na}
    e_map = {"low": cfg.entropy_low, "high": cfg.entropy_high, "extreme": cfg.entropy_extreme, "na": cfg.entropy_na}
    v_map = {"pos": cfg.velocity_pos, "neg": cfg.velocity_neg, "na": cfg.velocity_na}

    return {
        "hurst": h_map.get(h_c, cfg.hurst_na),
        "zscore": z_map.get(z_c, cfg.zscore_na),
        "entropy": e_map.get(e_c, cfg.entropy_na),
        "velocity": v_map.get(v_c, cfg.velocity_na),
    }


def sniper_mtf_bits_from_alignment_sentence(line: str) -> str | None:
    """Extrai P (Persistence), M (Mean Reversion), N (Noise) a partir do ALINHAMENTO."""
    parts: list[str] = []
    for seg in (line or "").split("|"):
        chunk = seg.strip()
        if ":" not in chunk:
            continue
        rhs = chunk.split(":", 1)[1].strip().lower()
        head = rhs.split()[0] if rhs else ""
        if head in ("persistence", "momentum", "bull", "bear", "trend"):
            parts.append("P")
        elif head in ("mean", "reversion", "reversao"):
            parts.append("M")
        elif head in ("noise", "random", "lateral"):
            parts.append("N")
        else:
            parts.append("?")
    if len(parts) >= 4:
        return "/".join(parts[:4])
    if len(parts) >= 3:
        return "/".join(parts[:3])
    return None


def format_sniper_prompt_line(
    symbol: str,
    macro_desc: str,
    structure_desc: str,
    swing_desc: str,
    trigger_desc: str,
    sniper_tokens: dict[str, str],
    sp: StrategyPayloadConfig | None = None,
    *,
    mtf_alignment_line: str = "",
) -> str:
    """Concatena tokens na ordem configurada no JSON da estrategia."""
    cfg = sp or DEFAULT_STRATEGY_PAYLOAD_CONFIG
    defaults = {"hurst": cfg.hurst_na, "zscore": cfg.zscore_na, "entropy": cfg.entropy_na, "velocity": cfg.velocity_na}
    parts: list[str] = []
    for key in cfg.payload_token_order:
        label = cfg.field_labels.get(key, key)
        val = sniper_tokens.get(key, defaults.get(key, cfg.hurst_na))
        parts.append(f"{label}{cfg.kv_separator}{val}")
    bits = sniper_mtf_bits_from_alignment_sentence(mtf_alignment_line)
    if bits is None:
        bits = "/".join(
            (
                _mtf_letter(macro_desc),
                _mtf_letter(structure_desc),
                _mtf_letter(swing_desc),
                _mtf_letter(trigger_desc),
            )
        )
    parts.append(f"{cfg.mtf_token_key}{cfg.kv_separator}{bits}")
    parts.append(f"{cfg.sym_token_key}{cfg.kv_separator}{symbol}")
    return cfg.pair_separator.join(parts)


def _mtf_letter(desc: str) -> str:
    """Codifica narrativa de regime em P (Persistence), M (Mean Reversion), N (Noise)."""
    d = (desc or "").lower()
    if "mean" in d or "reversion" in d:
        return "M"
    if "momentum" in d or "persistence" in d or "alpha" in d:
        return "P"
    if "random" in d or "noise" in d or "ruido" in d:
        return "N"
    return "?"
