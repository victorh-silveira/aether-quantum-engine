"""Deteccao de colapso de classe durante o treino temporal."""

from typing import Any


def _pred_side_skew(*, label_call_frac: float | None, pred_call_frac: float | None, bias_cap: float) -> bool:
    """Detecta predicoes concentradas em um lado."""
    if pred_call_frac is None:
        return False
    prediction = float(pred_call_frac)
    if abs(prediction - 0.5) > bias_cap:
        return True
    return label_call_frac is not None and abs(prediction - float(label_call_frac)) > bias_cap


def _majority_collapse_hit(
    gate_cfg: dict[str, Any],
    *,
    label_call_frac: float | None,
    pred_call_frac: float | None,
    minority_recall: float | None,
) -> bool:
    """Marca colapso preditivo ou classe minoritaria sem recall minimo."""
    if not bool(gate_cfg.get("reject_majority_collapse", False)):
        return False
    bias_cap = float(gate_cfg.get("max_label_call_frac_bias", 0.20))
    min_recall = float(gate_cfg.get("min_minority_recall", 0.25))
    if _pred_side_skew(label_call_frac=label_call_frac, pred_call_frac=pred_call_frac, bias_cap=bias_cap):
        return True
    if minority_recall is None or float(minority_recall) + 1e-9 >= min_recall:
        return False
    return label_call_frac is not None and abs(float(label_call_frac) - 0.5) > bias_cap
