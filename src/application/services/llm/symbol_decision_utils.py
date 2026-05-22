"""Utilitarios de processamento para decisoes de simbolos via LLM."""

from __future__ import annotations

from typing import Any

from src.application.services.llm import (
    dual_confluence_prompt_fragment,
    format_numeric_indicators_six_line,
)
from src.application.services.llm.context_runtime import fetch_context_blocks
from src.application.services.llm.global_macro_confluence import (
    MacroSnapshot,
    reconcile_cluster_tags_with_macro,
    resolve_macro_config,
)
from src.application.services.llm.llm_bridge_guards import apply_macro_confluence_guard
from src.application.services.llm.macro_snapshot_fetch import fetch_macro_snapshot
from src.application.services.llm.prompt_extras import build_institutional_pa_bundle
from src.application.services.llm.prompt_utils import (
    build_sniper_trading_prompt as _build_prompt_impl,
)
from src.application.services.llm.sniper_payload import coerce_sniper_tokens
from src.domain.models.trade import TradeDirection


def decision_from_payload(
    payload: dict[str, Any],
) -> tuple[TradeDirection | None, float, str, TradeDirection | None, TradeDirection | None]:
    """Converte payload padronizado em direcao, conviccao e nota."""
    tag = payload.get("_direction_normalized")
    note = " ".join(str(payload.get("note", "")).replace("\n", " ").split()).strip() or "-"

    if tag == "CALL":
        direction = TradeDirection.CALL
    elif tag == "PUT":
        direction = TradeDirection.PUT
    else:
        direction = None

    conviction = float(payload.get("_conviction_normalized", 0.0))

    us_dir_str = str(payload.get("us_cluster", "")).upper()
    eu_dir_str = str(payload.get("eu_cluster", "")).upper()

    us_dir = TradeDirection.CALL if us_dir_str == "CALL" else (TradeDirection.PUT if us_dir_str == "PUT" else None)
    eu_dir = TradeDirection.CALL if eu_dir_str == "CALL" else (TradeDirection.PUT if eu_dir_str == "PUT" else None)

    return direction, conviction, note, us_dir, eu_dir


def anchor_llm_decision_complete(
    orch: Any,
    sym: str,
    direction: TradeDirection | None,
    us_dir: TradeDirection | None,
    eu_dir: TradeDirection | None,
    macro_snapshot: MacroSnapshot | None = None,
) -> tuple[bool, str]:
    """Exige EURUSD e tags de cluster CALL/PUT da LLM quando correlacao esta ativa."""
    anchor = str(getattr(orch, "anchor", sym) or sym)
    if sym != anchor:
        return True, ""
    if direction not in (TradeDirection.CALL, TradeDirection.PUT):
        return False, "LLM_EURUSD_AUSENTE"
    strategy = orch.config.get("strategy", {}) if isinstance(getattr(orch, "config", None), dict) else {}
    corr = strategy.get("correlation", {}) if isinstance(strategy.get("correlation"), dict) else {}
    if not bool(corr.get("enabled", False)):
        return True, ""
    us_quant_flat = macro_snapshot is not None and macro_snapshot.us_dir == "flat"
    eu_quant_flat = macro_snapshot is not None and macro_snapshot.eu_dir == "flat"
    if not us_quant_flat and us_dir not in (TradeDirection.CALL, TradeDirection.PUT):
        return False, "LLM_US_CLUSTER_AUSENTE"
    if not eu_quant_flat and eu_dir not in (TradeDirection.CALL, TradeDirection.PUT):
        return False, "LLM_EU_CLUSTER_AUSENTE"
    return True, ""


def apply_macro_post_parse(
    direction: TradeDirection | None,
    conviction: float,
    note: str,
    us_dir: TradeDirection | None,
    eu_dir: TradeDirection | None,
    macro_snapshot: MacroSnapshot,
    macro_cfg: dict[str, Any] | None,
    sym: str | None = None,
) -> tuple[TradeDirection | None, float, str, TradeDirection | None, TradeDirection | None, bool, bool]:
    """Aplica guard macro na decisao EURUSD apos parse da LLM."""
    us_name = us_dir.name if us_dir in (TradeDirection.CALL, TradeDirection.PUT) else None
    eu_name = eu_dir.name if eu_dir in (TradeDirection.CALL, TradeDirection.PUT) else None
    us_aligned, eu_aligned, cluster_changed, cluster_note = reconcile_cluster_tags_with_macro(
        us_name,
        eu_name,
        macro_snapshot,
        macro_cfg,
    )
    if cluster_changed:
        us_dir = TradeDirection[us_aligned] if us_aligned in ("CALL", "PUT") else None
        eu_dir = TradeDirection[eu_aligned] if eu_aligned in ("CALL", "PUT") else None
        note = f"{note} | {cluster_note}".strip()
    direction, conviction, macro_guard, macro_note, macro_execute = apply_macro_confluence_guard(
        direction,
        conviction,
        macro_snapshot,
        macro_cfg,
        sym=sym,
    )
    if macro_note:
        note = f"{note} | {macro_note}".strip()
    macro_guard = macro_guard or cluster_changed
    return direction, conviction, note, us_dir, eu_dir, macro_guard, macro_execute


async def build_symbol_prompt(
    orch: Any,
    sym: str,
    runtime: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], float, str, str, str, str, str, str, str, list[float], MacroSnapshot]:
    """Extrai contexto e constroi o prompt final para a LLM."""
    macro_snapshot = await fetch_macro_snapshot(orch, runtime)
    macro_d, struct_d, swing_d, trig_d, mtf_d, ctx = await fetch_context_blocks(orch, sym, runtime)
    ctx["macro_confluence"] = macro_snapshot.macro_block
    ctx["fx_reference_line"] = macro_snapshot.fx_reference_line
    ctx["macro_sentiment"] = macro_snapshot.tag
    ctx["eurusd_bias_quant"] = macro_snapshot.eurusd_bias
    ctx["macro_us_dir"] = macro_snapshot.us_dir
    ctx["macro_eu_dir"] = macro_snapshot.eu_dir
    regime_label = str(ctx.get("regime_label", "range"))
    line_macro_structure = str(ctx.get("llm_mtf_confluence_m30_m5") or "")
    line_swing_trigger = str(ctx.get("llm_mtf_confluence") or "")
    macro_c = list(ctx.get("llm_macro_closes") or ctx.get("m15_closes") or [])
    structure_c = list(ctx.get("llm_structure_closes") or [])
    swing_c = list(ctx.get("llm_swing_closes") or ctx.get("m5_closes") or [])
    trigger_c = list(ctx.get("llm_trigger_closes") or ctx.get("m3_closes") or [])
    micro_swing_c = list(ctx.get("llm_micro_swing_closes") or [])
    micro_trigger_c = list(ctx.get("llm_micro_trigger_closes") or [])
    tf_tags = ctx.get("llm_tf_numeric_tags")
    if not isinstance(tf_tags, tuple) or len(tf_tags) != 6:
        ic = runtime["indicator_config"]
        tf_tags = tuple(ic.tf_tags) if len(ic.tf_tags) == 6 else ("1440", "240", "60", "15", "5", "1")

    indicators_numeric_line = format_numeric_indicators_six_line(
        macro_c,
        structure_c,
        swing_c,
        trigger_c,
        micro_swing_c,
        micro_trigger_c,
        runtime["indicator_config"],
        ctx.get("entropy_swing"),
        ctx.get("vol_range_pct"),
        mtf_d,
        line_swing_trigger,
        tf_tags=tf_tags,
        ema_guard=str(ctx.get("ema_guard", "")),
    )
    entropy_swing = ctx.get("entropy_swing")
    cf_dual = dual_confluence_prompt_fragment(line_macro_structure, line_swing_trigger)

    institutional_pa_bundle = build_institutional_pa_bundle(
        regime_label=regime_label,
        entropy_swing=float(entropy_swing) if entropy_swing is not None else None,
        vol_range_pct=float(ctx["vol_range_pct"]) if ctx.get("vol_range_pct") is not None else None,
        indicators_numeric_line=indicators_numeric_line,
        cf_dual=cf_dual,
        line_macro_structure=line_macro_structure,
        line_swing_trigger=line_swing_trigger,
        ema_guard=str(ctx.get("ema_guard", "")),
        compact=False,
    )
    sniper_tok = coerce_sniper_tokens(ctx.get("sniper_tokens"))
    bundle_txt = str(ctx.get("llm_indicator_bundle") or "")
    wr_v, wr_n = None, 0
    rm = getattr(orch, "risk_manager", None)
    if rm is not None and hasattr(rm, "get_wr_rolling_stats"):
        raw_wr = rm.get_wr_rolling_stats(sym)
        if isinstance(raw_wr, tuple) and len(raw_wr) == 2:
            wr_v, wr_n = raw_wr[0], int(raw_wr[1])

    cluster_status = macro_snapshot.cluster_status
    tf_labels = ctx.get("llm_tf_labels")
    if not isinstance(tf_labels, tuple) or len(tf_labels) != 6:
        tf_labels = ("D1", "H4", "H1", "M15", "M5", "M1")

    strategy_cfg = orch.config.get("strategy", {}) if hasattr(orch, "config") and isinstance(orch.config, dict) else {}
    macro_cfg = resolve_macro_config(strategy_cfg.get("macro"))

    statarb_z = (
        float(macro_snapshot.statarb_spreads[sym])
        if macro_snapshot.statarb_spreads and sym in macro_snapshot.statarb_spreads
        else None
    )
    hmm_state = int(macro_snapshot.hmm_state) if hasattr(macro_snapshot, "hmm_state") else None
    hmm_prob = float(macro_snapshot.hmm_prob) if hasattr(macro_snapshot, "hmm_prob") else None

    prompt = _build_prompt_impl(
        sym,
        macro_d,
        struct_d,
        swing_d,
        trig_d,
        sniper_tok,
        mtf_d,
        str(ctx.get("regime_line", "")),
        str(ctx.get("session_line", "")),
        str(ctx.get("micro_line", "")),
        trigger_c,
        runtime["payout_estimate"],
        runtime["min_payout_accept"],
        runtime["duration"],
        runtime["du"],
        trigger_ohlc=ctx.get("llm_trigger_ohlc"),
        strategy_payload=runtime.get("strategy_payload"),
        institutional_pa_bundle=institutional_pa_bundle,
        indicator_bundle_line=bundle_txt,
        wr_rolling=wr_v,
        wr_samples=wr_n,
        cluster_status=cluster_status,
        micro_swing_desc=str(ctx.get("micro_swing_desc", "")),
        micro_trigger_desc=str(ctx.get("micro_trigger_desc", "")),
        mtf_matrix=str(ctx.get("mtf_matrix", "")),
        tf_labels=tf_labels,
        metrics_h=ctx.get("hurst_value"),
        metrics_z=ctx.get("zscore_value"),
        metrics_e=ctx.get("entropy_trigger"),
        macro_confluence=macro_snapshot.macro_block,
        fx_reference_line=macro_snapshot.fx_reference_line,
        macro_sentiment=macro_snapshot.tag,
        statarb_z=statarb_z,
        hmm_state=hmm_state,
        hmm_prob=hmm_prob,
    )
    ctx["macro_cfg"] = macro_cfg
    return (
        prompt,
        ctx,
        sniper_tok,
        0.5,
        indicators_numeric_line,
        institutional_pa_bundle,
        macro_d,
        struct_d,
        swing_d,
        trig_d,
        mtf_d,
        swing_c,
        macro_snapshot,
    )
