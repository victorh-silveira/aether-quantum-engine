"""Pipeline de regime universal: avaliacao, veto de inversao DL e coesao de compressao micro."""

from __future__ import annotations

from src.application.services.execution_direction_compression_trap import enforce_compression_trap_micro_bb_cohesion
from src.application.services.execution_direction_inversion_veto import veto_inversion_on_dl_conviction
from src.application.services.execution_universal_regime_evaluator import UniversalRegimeEvaluator
from src.domain.models.trade import TradeDirection


def apply_universal_regime_pipeline(
    metrics: dict,
    *,
    exec_dir: TradeDirection,
    dl_dir: TradeDirection,
    cfg: dict,
    mandatory_floor: float,
    recovery_active: bool,
    dl_inversion_veto_score: float,
    neutral_token: str,
) -> TradeDirection:
    """Classifica regime M15, veta inversao de alta conviccao DL e confirma coesao micro M1."""
    continuous_mode = bool(cfg.get("mandatory_trade_each_cycle", False))
    kelly_cfg = cfg.get("kelly") if isinstance(cfg.get("kelly"), dict) else {}
    regime_cfg = cfg.get("regime_evaluator") if isinstance(cfg.get("regime_evaluator"), dict) else {}
    evaluator = UniversalRegimeEvaluator(
        regime_cfg,
        recovery_active=recovery_active,
        continuous_mode=continuous_mode,
        mandatory_min_signal=mandatory_floor,
        kelly_cfg=kelly_cfg,
    )
    regime_eval = evaluator.evaluate(metrics, dl_dir=dl_dir, exec_dir=exec_dir)
    regime_eval = veto_inversion_on_dl_conviction(regime_eval, metrics, dl_dir, veto_score=dl_inversion_veto_score)
    resolved = evaluator.apply(metrics, regime_eval, exec_dir, dl_dir=dl_dir)
    if regime_eval.regime is None and not metrics.get("universal_regime"):
        metrics["universal_regime"] = neutral_token
        metrics["universal_regime_scenario"] = neutral_token
    return enforce_compression_trap_micro_bb_cohesion(resolved, dl_dir, metrics, regime_eval)
