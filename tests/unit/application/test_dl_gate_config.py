"""Teste da deteccao de colapso de classe no treino."""

from src.application.services.deep_learning.dl_gate_config import _majority_collapse_hit, _pred_side_skew


def test_pred_side_skew_requires_prediction():
    assert _pred_side_skew(label_call_frac=0.5, pred_call_frac=None, bias_cap=0.15) is False
    assert _pred_side_skew(label_call_frac=0.5, pred_call_frac=0.8, bias_cap=0.15) is True
    assert _pred_side_skew(label_call_frac=0.3, pred_call_frac=0.5, bias_cap=0.15) is True


def test_majority_collapse_uses_training_quality():
    cfg = {"reject_majority_collapse": True, "max_label_call_frac_bias": 0.15, "min_minority_recall": 0.45}
    assert _majority_collapse_hit(cfg, label_call_frac=0.5, pred_call_frac=0.8, minority_recall=0.2)
    assert not _majority_collapse_hit(cfg, label_call_frac=0.5, pred_call_frac=0.5, minority_recall=0.8)
    assert not _majority_collapse_hit({}, label_call_frac=0.5, pred_call_frac=0.8, minority_recall=0.2)
    assert _majority_collapse_hit(cfg, label_call_frac=0.3, pred_call_frac=0.44, minority_recall=0.1)
