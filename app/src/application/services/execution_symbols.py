"""Simbolos elegiveis e ranking de candidatos para execucao."""

from src.application.services.deep_learning.dl_gating import calibration_gap
from src.application.services.deep_learning.dl_post_loss import post_loss_block_reason
from src.domain.models.trade import TradeDirection


_DEFAULT_SELECTION = {
    "min_conviction_execute": 0.56,
    "min_edge_margin": 0.06,
    "min_val_accuracy": 0.50,
    "strong_raw": 0.65,
    "strong_edge": 0.12,
    "max_calib_gap": 0.18,
    "min_raw": 0.52,
    "max_raw_saturation": 0.97,
    "saturation_min_trade_score": 0.58,
}


def symbols_eligible_for_execution(anchor: str, symbols: list[str], *, include_anchor: bool) -> list[str]:
    """Retorna simbolos do cluster que podem receber ordens de execucao."""
    eligible = []
    for symbol in symbols:
        if symbol == anchor and not include_anchor:
            continue
        eligible.append(symbol)
    return eligible


def _trade_score(metrics: dict) -> float:
    """Le score unificado de conviccao usado em selecao e ranking."""
    return float(metrics.get("trade_score", metrics.get("conviction", 0.0)))


def _calib_gap_penalty(metrics: dict, cfg: dict) -> float:
    """Penaliza ranking quando o gap entre score calibrado e raw excede o limite."""
    score = _trade_score(metrics)
    raw = metrics.get("raw_prob")
    if raw is None:
        return 0.0
    gap = calibration_gap(score, float(raw))
    limit = float(cfg.get("max_calib_gap", 0.18))
    if gap <= limit:
        return 0.0
    return min(0.15, (gap - limit) * 0.5)


def candidate_execution_score(
    metrics: dict,
    *,
    recovery_active: bool,
    selection: dict | None = None,
) -> float:
    """Pontua candidato com score calibrado unificado, val_acc e edge."""
    cfg = {**_DEFAULT_SELECTION, **(selection or {})}
    score = _trade_score(metrics)
    val = float(metrics.get("val_accuracy", 0.0))
    edge = float(metrics.get("edge", abs(score - 0.5)))
    penalty = _calib_gap_penalty(metrics, cfg)
    if recovery_active:
        return score * 0.35 + val * 0.50 + edge * 0.15 - penalty
    return score * 0.40 + val * 0.45 + edge * 0.15 - penalty


def _selection_hard_reject(metrics: dict, cfg: dict, score: float, raw_side: float) -> bool:
    """Indica rejeicao imediata por gap, raw fraco, live_wr ou saturacao."""
    max_gap = float(cfg.get("max_calib_gap", 0.18))
    min_raw = float(cfg.get("min_raw", 0.52))
    gap_fail = (
        "raw_prob" in metrics
        and calibration_gap(score, float(metrics["raw_prob"])) > max_gap + 1e-9
        and float(metrics.get("val_accuracy", 0.0)) + 1e-9 < 0.65
    )
    live_wr = metrics.get("live_win_rate")
    live_fail = live_wr is not None and float(live_wr) + 1e-9 < 0.42
    sat_min_score = float(cfg.get("saturation_min_trade_score", 0.58))
    sat_max_raw = float(cfg.get("max_raw_saturation", 0.97))
    sat_fail = "raw_prob" in metrics and raw_side + 1e-9 > sat_max_raw and score + 1e-9 >= sat_min_score
    return gap_fail or raw_side + 1e-9 < min_raw or live_fail or sat_fail


def _passes_selection_gate(metrics: dict, cfg: dict) -> bool:
    """Verifica se metricas do candidato passam limiares de selecao de precisao."""
    score = _trade_score(metrics)
    val = float(metrics.get("val_accuracy", 0.0))
    edge = float(metrics.get("edge", abs(score - 0.5)))
    raw_side = float(metrics.get("raw_conviction", metrics.get("raw_prob", score)))
    if "raw_prob" in metrics:
        raw_side = max(float(metrics["raw_prob"]), 1.0 - float(metrics["raw_prob"]))
    min_conv = float(cfg["min_conviction_execute"])
    min_edge = float(cfg["min_edge_margin"])
    min_val = float(cfg["min_val_accuracy"])
    strong_raw = float(cfg["strong_raw"])
    strong_edge = float(cfg["strong_edge"])
    if _selection_hard_reject(metrics, cfg, score, raw_side):
        return False
    if score + 1e-9 >= min_conv and edge + 1e-9 >= min_edge:
        return True
    if val + 1e-9 >= min_val:
        return True
    return raw_side + 1e-9 >= strong_raw and edge + 1e-9 >= strong_edge


def filter_execution_candidates(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    selection: dict | None = None,
) -> list[tuple[str, TradeDirection, dict]]:
    """Mantem candidatos que passam os mesmos limiares do gating de precisao."""
    cfg = {**_DEFAULT_SELECTION, **(selection or {})}
    return [item for item in candidates if _passes_selection_gate(item[2], cfg)]


def select_best_execution_candidate(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    diversify_margin: float,
    recovery_active: bool,
) -> tuple[str, TradeDirection, dict]:
    """Escolhe melhor candidato por score e evita repetir simbolo da ultima loss."""
    pool = list(candidates)
    if recovery_active and last_loss_symbol:
        filtered = [item for item in pool if item[0] != last_loss_symbol]
        if filtered:
            pool = filtered
    ranked = sorted(
        pool,
        key=lambda item: candidate_execution_score(item[2], recovery_active=recovery_active),
        reverse=True,
    )
    best = ranked[0]
    if len(ranked) >= 2 and last_loss_symbol and best[0] == last_loss_symbol and not recovery_active:
        top_score = candidate_execution_score(best[2], recovery_active=recovery_active)
        alt_score = candidate_execution_score(ranked[1][2], recovery_active=recovery_active)
        if top_score - alt_score <= diversify_margin:
            return ranked[1]
    return best


def filter_post_loss_banned_candidates(
    orch,
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    flip_raw_min: float,
) -> list[tuple[str, TradeDirection, dict]]:
    """Remove candidatos ainda vetados por post_loss na combinacao simbolo+direcao."""
    kept: list[tuple[str, TradeDirection, dict]] = []
    for symbol, direction, metrics in candidates:
        raw = metrics.get("raw_prob")
        raw_prob = float(raw) if raw is not None else None
        if post_loss_block_reason(orch, symbol, direction, raw_prob=raw_prob, flip_raw_min=flip_raw_min):
            continue
        kept.append((symbol, direction, metrics))
    return kept


def select_mandatory_execution_candidate(
    orch,
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    diversify_margin: float,
    recovery_active: bool,
    flip_raw_min: float,
) -> tuple[str, TradeDirection, dict]:
    """Escolhe candidato em modo obrigatorio respeitando post_loss e preferindo execute=true."""
    unbanned = filter_post_loss_banned_candidates(orch, candidates, flip_raw_min=flip_raw_min)
    pool = unbanned if unbanned else list(candidates)
    approved = [item for item in pool if item[2].get("execute")]
    ranking_pool = approved if approved else pool
    return select_best_execution_candidate(
        ranking_pool,
        last_loss_symbol=last_loss_symbol,
        diversify_margin=diversify_margin,
        recovery_active=recovery_active,
    )


def pending_recovery_active(pending_loss: dict) -> bool:
    """Indica se ha perda pendente ativando modo de recuperacao na selecao."""
    return sum(float(v) for v in pending_loss.values()) > 0.0


def format_execution_alternates(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    exclude_symbol: str | None = None,
    limit: int = 2,
) -> str:
    """Formata simbolos alternativos ordenados por score para log EXEC_SEL."""
    ranked = sorted(
        candidates,
        key=lambda item: candidate_execution_score(item[2], recovery_active=False),
        reverse=True,
    )
    alts = [item for item in ranked if item[0] != exclude_symbol][:limit]
    return ", ".join(
        f"{symbol}({metrics.get('trade_score', metrics.get('conviction', 0.0)):.2f})" for symbol, _, metrics in alts
    )
