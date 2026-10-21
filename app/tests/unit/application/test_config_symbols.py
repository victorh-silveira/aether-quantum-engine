from src.application.services.orchestrator.config_symbols import normalize_symbols_and_anchor, resolve_dl_train_symbols
from tests.market_symbols import ALT_SYMBOL, ANCHOR


def test_normalize_symbols_and_anchor_inserts_anchor_when_absent():
    anchor, symbols = normalize_symbols_and_anchor({"anchor": ANCHOR, "symbols": [ALT_SYMBOL]})
    assert anchor == ANCHOR
    assert symbols[0] == ANCHOR
    assert ALT_SYMBOL in symbols


def test_resolve_dl_train_symbols_defaults_to_active_symbols():
    config = {"anchor": ANCHOR, "symbols": [ANCHOR, ALT_SYMBOL]}
    assert resolve_dl_train_symbols(config) == [ANCHOR, ALT_SYMBOL]


def test_resolve_dl_train_symbols_honors_explicit_list():
    config = {
        "anchor": ANCHOR,
        "symbols": [ANCHOR, ALT_SYMBOL],
        "deep_learning": {"train_symbols": [ALT_SYMBOL]},
    }
    assert resolve_dl_train_symbols(config) == [ALT_SYMBOL]


def test_resolve_dl_train_symbols_empty_list_skips_training():
    config = {"anchor": ANCHOR, "symbols": [ANCHOR, ALT_SYMBOL], "deep_learning": {"train_symbols": []}}
    assert resolve_dl_train_symbols(config) == []


def test_explicit_training_symbols_include_approved_runtime_discovery(tmp_path, monkeypatch):
    import json

    from src.application.services.orchestrator import config_symbols

    models = tmp_path / "indicator/data/indicator-models"
    models.mkdir(parents=True)
    (models / "R_50_300.json").write_text(
        json.dumps(
            {
                "symbol": "R_50",
                "period": 300,
                "status": "aprovado",
                "validation_accuracy": 0.55,
                "baseline_accuracy": 0.5,
                "validation_brier": 0.2,
                "baseline_brier": 0.25,
            }
        )
    )
    monkeypatch.setattr(config_symbols, "_REPOSITORY_ROOT", tmp_path)
    config = {
        "anchor": "R_25",
        "symbols": ["R_25"],
        "deep_learning": {"train_symbols": ["R_10", "R_25", "R_10"]},
        "indicator_consensus": {"enabled": True, "auto_symbols": True},
    }
    assert resolve_dl_train_symbols(config) == ["R_10", "R_25", "R_50"]
    assert normalize_symbols_and_anchor(config) == ("R_25", ["R_25", "R_50"])
    config["pullback_strategy"] = {"enabled": True}
    assert normalize_symbols_and_anchor(config) == ("R_25", ["R_25"])


def test_approved_indicator_symbols_accepts_bear_and_bull(tmp_path):
    import json

    from src.application.services.orchestrator.config_symbols import approved_indicator_symbols

    models = tmp_path / "indicator/data/indicator-models"
    models.mkdir(parents=True)
    for sym in ("RDBEAR", "RDBULL", "INVALID_SYM"):
        (models / f"{sym}_300.json").write_text(
            json.dumps(
                {
                    "symbol": sym,
                    "period": 300,
                    "status": "aprovado",
                    "validation_accuracy": 0.60,
                    "baseline_accuracy": 0.50,
                    "validation_brier": 0.20,
                    "baseline_brier": 0.25,
                }
            )
        )
    assert approved_indicator_symbols(models) == ["RDBEAR", "RDBULL"]
