"""Selecao e calibracao em blocos temporais separados."""

from indicator.research import brier, calibrate, logits, select_and_calibrate


def test_candidate_selection_does_not_see_validation_labels():
    fit_rows = [(0.1 * ((i % 5) - 2),) * 8 for i in range(100)]
    fit_labels = [int(i % 5 >= 3) for i in range(100)]
    select_rows = fit_rows[:30]
    select_labels = fit_labels[:30]
    cal_rows = fit_rows[30:60]
    cal_labels = fit_labels[30:60]
    result = select_and_calibrate(fit_rows, fit_labels, select_rows, select_labels, cal_rows, cal_labels)
    assert len(result[0]) == 8
    assert 0.0 <= result[4] <= 5.0
    assert len(result[6]) == 2
    assert min(score for _, score in result[6]) >= 0.0
    assert result[3] in (0.001, 0.05)
    assert len(logits(select_rows, result[0], result[1])) == 30


def test_calibration_never_worsens_its_own_block():
    scores = [-1.0, -0.5, 0.5, 1.0] * 10
    labels = [0, 0, 1, 1] * 10
    slope, offset = calibrate(scores, labels)
    assert 0.0 <= slope <= 5.0
    assert brier(scores, labels, slope, offset) <= brier(scores, labels)
