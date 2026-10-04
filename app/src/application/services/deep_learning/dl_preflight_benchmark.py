"""Benchmark linear pre-flight para diagnostico de separabilidade do sinal."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


logger = logging.getLogger("AETH")


def run_preflight_linear_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float]:
    """Calcula acuracia temporal de DummyClassifier e LogisticRegression na validacao."""
    if len(x_train) < 20 or len(x_val) < 5:
        return 0.5, 0.5
    unique_train = np.unique(y_train)
    if len(unique_train) < 2:
        return 0.5, 0.5
    x_tr_flat = x_train.reshape(len(x_train), -1)
    x_val_flat = x_val.reshape(len(x_val), -1)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_tr_flat, y_train)
    acc_dummy = float(accuracy_score(y_val, dummy.predict(x_val_flat)))

    clf = LogisticRegression(C=0.01, max_iter=200, solver="liblinear", random_state=42)
    clf.fit(x_tr_flat, y_train)
    acc_linear = float(accuracy_score(y_val, clf.predict(x_val_flat)))

    val_call_pct = float(np.mean(y_val)) * 100.0 if len(y_val) else 50.0
    val_majority_pct = max(val_call_pct, 100.0 - val_call_pct)
    logger.info(
        "DL PRE-FLIGHT BENCHMARK | Dummy ACC: %.1f%% | LogisticRegression val ACC: %.1f%% | "
        "Maioria val: %.1f%% | Validacao CALL: %.1f%% | delta_maioria=%+.1f pp",
        acc_dummy * 100.0,
        acc_linear * 100.0,
        val_majority_pct,
        val_call_pct,
        acc_linear * 100.0 - val_majority_pct,
    )
    return acc_dummy, acc_linear
