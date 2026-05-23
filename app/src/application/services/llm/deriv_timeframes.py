"""Rotulos e tags compactas para granularidades da Deriv (segundos por vela)."""

from __future__ import annotations


_DERIV_TF_LABELS: dict[int, str] = {
    86400: "D1",
    14400: "H4",
    3600: "H1",
    1800: "M30",
    900: "M15",
    300: "M5",
    180: "M3",
    60: "M1",
}


def deriv_tf_label(granularity_seconds: int) -> str:
    """Rotulo estavel H1, M15, M5, M1 etc. para texto de prompt e alinhamento MTF."""
    g = int(granularity_seconds)
    if g in _DERIV_TF_LABELS:
        return _DERIV_TF_LABELS[g]
    if g <= 0:
        return "TF?"
    m = g // 60
    if m <= 0:
        return f"s{g}"
    if m < 60:
        return f"M{m}"
    if m % 60 == 0:
        h = m // 60
        return f"H{h}" if h <= 24 else f"{m}m"
    return f"{m}m"


def deriv_tf_compact_numeric_tag(granularity_seconds: int) -> str:
    """Tag numerica em minutos por vela para LLM_DADOS_NUM (ex.: H1->60, M15->15, M1->1)."""
    g = int(granularity_seconds)
    if g <= 0:
        return "?"
    return str(max(1, g // 60))
