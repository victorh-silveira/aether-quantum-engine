"""Contrato operacional Rise/Fall: launchers, SSOT e features direcionais."""

from pathlib import Path

from src.application.services.deep_learning.dl_feature_orthogonal import FEATURE_DIM, ORTHOGONAL_FEATURE_NAMES
from src.application.services.strategy.decision_mode import resolve_decision_mode
from src.domain.config_knobs import load_settings_json
from src.domain.models.trade import TradeDirection
from src.infrastructure.handlers.trade_handler import build_proposal_request


def test_rise_fall_settings_and_indicators_are_consistent():
    settings = load_settings_json()
    assert resolve_decision_mode(settings) == "deep_learning"
    assert "touch" not in settings
    assert "barrier_contracts" not in settings["risk_management"]
    assert settings["risk_management"]["params"]["contract_type"] == "CALL"
    assert settings["risk_management"]["params"]["duration"] == 5
    assert settings["risk_management"]["params"]["duration_unit"] == "m"
    assert settings["deep_learning"]["label_horizon_bars"] == 1
    assert settings["data_handler"]["micro_granularity"] == 300
    assert FEATURE_DIM == len(ORTHOGONAL_FEATURE_NAMES) == 14
    for feature in ("rsi_centered", "norm_atr", "macd_hist_norm", "adx_scaled", "hurst_centered"):
        assert feature in ORTHOGONAL_FEATURE_NAMES
    for direction in (TradeDirection.CALL, TradeDirection.PUT):
        proposal = build_proposal_request("1HZ75V", direction, 1.0, settings["risk_management"]["params"])
        assert proposal["contract_type"] == direction.value
        assert "barrier" not in proposal


def test_rise_fall_launchers_train_tcn_and_meta_without_candidate_promotion():
    root = Path(__file__).resolve().parents[4]
    launchers = (
        root / "app" / "scripts" / "batch" / "launch-train.bat",
        root / "app" / "scripts" / "wsl" / "launch-train-wsl.sh",
    )
    for path in launchers:
        script = path.read_text(encoding="utf-8")
        assert "run_launch_train_tf_pipeline.py" in script
        assert "train_meta_classifier.py" in script
        assert "train_loss_classifier" in script
        assert "check_dl_deploy_gate.py" in script
        assert "train_touch_classifier.py" not in script
        assert "meta_candidate.joblib" not in script
        assert "Meta exportado em meta-models" in script
        assert "Meta nao exportado; somente candidato diagnostico" in script
        assert "Nenhum candidato meta foi promovido" not in script


def test_windows_launcher_echo_does_not_pipe_to_an_unintended_command():
    root = Path(__file__).resolve().parents[4]
    script = (root / "app" / "scripts" / "batch" / "launch-train.bat").read_text(encoding="utf-8")
    for line in script.splitlines():
        if line.lstrip().lower().startswith("echo "):
            assert "|" not in line, "Pipe nao escapado em echo de batch executa o texto seguinte como comando"
