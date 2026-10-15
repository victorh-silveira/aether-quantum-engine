"""Testes unitarios para ranqueamento e promocao do setup pullback M5."""

import json
from pathlib import Path

from indicator.pullback_promotion import format_ranking_table, promote_best_m5_pair, rank_m5_pairs


def test_rank_m5_pairs_orders_by_ev_and_win_rate():
    report = {
        "R_25:300": {
            "estado": "PROXY DE VELAS",
            "setups": 20,
            "acertos": 12,
            "erros": 8,
            "empates": 0,
        },
        "1HZ25V:300": {
            "estado": "PROXY DE VELAS",
            "setups": 30,
            "acertos": 21,
            "erros": 9,
            "empates": 0,
        },
        "R_50:300": {
            "estado": "SEM SINAL",
            "setups": 0,
        },
        "R_25:900": {
            "estado": "PROXY DE VELAS",
            "setups": 15,
            "acertos": 10,
            "erros": 5,
        },
        "1HZ10V:300": {
            "estado": "PROXY DE VELAS",
            "setups": 0,
            "acertos": 0,
            "erros": 0,
        },
    }
    excluded = {"R_100"}
    ranked = rank_m5_pairs(report, excluded_symbols=excluded, payout=0.85)
    assert len(ranked) == 3
    assert ranked[0]["symbol"] == "1HZ25V"
    assert ranked[0]["eligible"] is True
    assert ranked[0]["win_rate"] == 0.70
    assert ranked[0]["ev"] > ranked[1]["ev"]

    excluded_ranked = rank_m5_pairs(report, excluded_symbols={"1HZ25V"}, payout=0.85)
    assert excluded_ranked[0]["symbol"] == "R_25"


def test_format_ranking_table():
    ranked = [
        {
            "symbol": "1HZ25V",
            "period": 300,
            "setups": 20,
            "wins": 14,
            "losses": 6,
            "ties": 0,
            "win_rate": 0.70,
            "ev": 0.295,
            "eligible": True,
            "excluded": False,
        },
        {
            "symbol": "R_50",
            "period": 300,
            "setups": 10,
            "wins": 4,
            "losses": 6,
            "ties": 0,
            "win_rate": 0.40,
            "ev": -0.26,
            "eligible": False,
            "excluded": True,
        },
    ]
    table = format_ranking_table(ranked)
    assert "RANKING DO SETUP PULLBACK M5" in table
    assert "1HZ25V" in table
    assert "PROMOVIDO" in table
    assert "EXCLUIDO" in table


def test_promote_best_m5_pair(tmp_path: Path):
    root = tmp_path
    models_dir = root / "indicator" / "data" / "indicator-models"
    models_dir.mkdir(parents=True)
    old_model = models_dir / "R_25_300.json"
    old_model.write_text(json.dumps({"symbol": "R_25", "period": 300, "status": "aprovado"}))
    same_model = models_dir / "1HZ25V_300.json"
    same_model.write_text(json.dumps({"symbol": "1HZ25V", "period": 300, "status": "candidato"}))
    corrupt_model = models_dir / "CORRUPT_300.json"
    corrupt_model.write_text("invalido")

    config_dir = root / "config"
    config_dir.mkdir(parents=True)
    settings_file = config_dir / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "anchor": "R_25",
                "symbols": ["R_25"],
                "pullback_strategy": {"enabled": True},
                "deep_learning": {"train_symbols": ["R_25"]},
            }
        )
    )

    drift_dir = root / "app" / "src" / "domain" / "symbols"
    drift_dir.mkdir(parents=True)

    best = {
        "symbol": "1HZ25V",
        "period": 300,
        "setups": 35,
        "wins": 24,
        "losses": 11,
        "ties": 0,
        "win_rate": 24.0 / 35.0,
        "ev": (24.0 / 35.0) * 0.85 - (11.0 / 35.0) * 1.0,
    }
    promote_best_m5_pair(best, root, variant="improved")

    new_model_file = models_dir / "1HZ25V_300.json"
    assert new_model_file.is_file()
    new_model = json.loads(new_model_file.read_text(encoding="utf-8"))
    assert new_model["status"] == "aprovado"
    assert new_model["symbol"] == "1HZ25V"

    old_content = json.loads(old_model.read_text(encoding="utf-8"))
    assert old_content["status"] == "candidato"

    cfg = json.loads(settings_file.read_text(encoding="utf-8"))
    assert cfg["anchor"] == "1HZ25V"
    assert cfg["symbols"] == ["1HZ25V"]
    assert cfg["pullback_strategy"]["promoted_symbol"] == "1HZ25V"
    assert "1HZ25V" in cfg["deep_learning"]["train_symbols"]

    drift_file = drift_dir / "drift_symbols.py"
    assert drift_file.is_file()
    drift_code = drift_file.read_text(encoding="utf-8")
    assert 'DEFAULT_ANCHOR = "1HZ25V"' in drift_code
    assert 'TRADING_SYMBOLS: tuple[str, ...] = ("1HZ25V",)' in drift_code
