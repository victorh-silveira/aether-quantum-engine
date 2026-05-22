"""Pos-processamento de decisao LLM por simbolo (entropia, conviccao, metricas)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.application.services.llm.global_macro_confluence import MacroSnapshot
from src.application.services.llm.regime import _shannon_entropy
from src.domain.models.trade import TradeDirection


def append_entropy_high_note(
    note: str,
    conviction: float,
    swing_c: list[float] | None,
    runtime: dict[str, Any],
    logger: Any,
) -> str:
    """Acrescenta marcador de entropia alta quando swing e conviccao exigem."""
    try:
        ic = runtime.get("indicator_config")
        if ic and swing_c:
            arr = np.array(swing_c, dtype=np.float64)
            ebins = int(getattr(ic, "entropy_bins", 30)) if hasattr(ic, "entropy_bins") else 30
            ewin = int(getattr(ic, "entropy_window", 20)) if hasattr(ic, "entropy_window") else 20
            entropy_val = _shannon_entropy(arr, ebins, ewin)
            if entropy_val > 3.0 and conviction > 0.75:
                return note + f" [ENTROPY_HIGH: {entropy_val:.2f}]"
    except Exception as e:
        if logger is not None:
            logger.debug("Erro na trava de entropia: %s", e)
    return note


def patch_final_symbol_metrics(
    metrics: dict[str, Any],
    *,
    execute_flag: bool,
    inverted: bool,
    llm_http_ms: float,
    llm_resp_chars: int,
    llm_direction_from_api: bool,
    us_dir: TradeDirection | None,
    eu_dir: TradeDirection | None,
    macro_snapshot: MacroSnapshot,
    macro_guard: bool,
) -> None:
    """Anexa telemetria final da decisao por simbolo."""
    metrics.update(
        {
            "execute": execute_flag,
            "llm_exec_inverted": inverted,
            "llm_http_ms": llm_http_ms,
            "llm_response_chars": llm_resp_chars,
            "llm_direction_from_api": llm_direction_from_api,
            "us_cluster": us_dir.name if us_dir else None,
            "eu_cluster": eu_dir.name if eu_dir else None,
            "entry_policy_tag": "",
            "macro_sentiment": macro_snapshot.tag,
            "macro_confluence_tag": macro_snapshot.tag,
            "eurusd_bias_quant": macro_snapshot.eurusd_bias,
            "macro_guard_applied": macro_guard,
            "macro_us_dir_quant": macro_snapshot.us_dir,
            "macro_eu_dir_quant": macro_snapshot.eu_dir,
            "macro_us_strength_quant": float(macro_snapshot.us_strength),
            "macro_eu_strength_quant": float(macro_snapshot.eu_strength),
            "statarb_spreads": dict(macro_snapshot.statarb_spreads or {}),
            "hmm_state": int(getattr(macro_snapshot, "hmm_state", 0)),
            "hmm_prob": float(getattr(macro_snapshot, "hmm_prob", 1.0)),
        }
    )
