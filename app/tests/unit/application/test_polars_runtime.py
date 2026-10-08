"""Contrato POLARS_MAX_THREADS no bootstrap."""

import os

from src.application.services.polars_runtime import DEFAULT_POLARS_MAX_THREADS, ensure_polars_max_threads


def test_ensure_polars_max_threads_setdefault(monkeypatch):
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    assert ensure_polars_max_threads() == DEFAULT_POLARS_MAX_THREADS
    assert os.environ["POLARS_MAX_THREADS"] == DEFAULT_POLARS_MAX_THREADS


def test_ensure_polars_max_threads_preserves_existing(monkeypatch):
    monkeypatch.setenv("POLARS_MAX_THREADS", "8")
    assert ensure_polars_max_threads() == "8"
