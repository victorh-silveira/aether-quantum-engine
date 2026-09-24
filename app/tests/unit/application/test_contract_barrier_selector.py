"""Testes unitarios para selecao e transicao de contratos de barreira (Touch / No-Touch)."""

from src.application.services.contract_barrier_selector import (
    load_barrier_config,
    resolve_barrier_offset,
    resolve_contract_barrier_structure,
    should_transition_to_barrier_contract,
)
from src.domain.models.contract_barrier_types import BarrierContractConfig
from src.domain.models.trade import TradeDirection


def test_load_barrier_config():
    """Verifica carregamento de configuracao de barreira a partir do dict global."""
    assert load_barrier_config(None).enabled is False
    assert load_barrier_config({}).enabled is False
    assert load_barrier_config({"risk_management": {}}).enabled is False

    cfg_dict = {
        "risk_management": {
            "barrier_contracts": {
                "enabled": True,
                "min_atr": 1.50,
                "target_regimes": ["explosion"],
                "barrier_multiplier": 0.75,
                "default_type": "NOTOUCH",
            }
        }
    }
    cfg = load_barrier_config(cfg_dict)
    assert cfg.enabled is True
    assert cfg.min_atr == 1.50
    assert cfg.target_regimes == ("explosion",)
    assert cfg.barrier_multiplier == 0.75
    assert cfg.default_type == "NOTOUCH"


def test_resolve_barrier_offset():
    """Verifica calculo e formatacao da barreira em funcao de direcao e ATR."""
    assert resolve_barrier_offset(1.20, TradeDirection.CALL, multiplier=0.80) == "+0.96"
    assert resolve_barrier_offset(1.20, TradeDirection.MULTUP, multiplier=0.80) == "+0.96"
    assert resolve_barrier_offset(1.20, TradeDirection.PUT, multiplier=0.80) == "-0.96"
    assert resolve_barrier_offset(1.20, TradeDirection.MULTDOWN, multiplier=0.80) == "-0.96"
    assert resolve_barrier_offset(0.0, TradeDirection.CALL, multiplier=0.0) == "+0.00"


def test_should_transition_to_barrier_contract():
    """Verifica logica booleana de transicao para contratos de barreira."""
    disabled_cfg = BarrierContractConfig(enabled=False)
    assert should_transition_to_barrier_contract({"atr": 2.0}, disabled_cfg) is False

    active_cfg = BarrierContractConfig(enabled=True, min_atr=1.20, target_regimes=("explosion",))
    assert should_transition_to_barrier_contract({"atr": 1.30}, active_cfg) is True
    assert should_transition_to_barrier_contract({"atr": 0.50, "regime": "vol_explosion_high"}, active_cfg) is True
    assert should_transition_to_barrier_contract({"atr": 0.50, "regime": "calm"}, active_cfg) is False
    assert should_transition_to_barrier_contract({"atr": "invalid"}, active_cfg) is False


def test_resolve_contract_barrier_structure():
    """Verifica enriquecimento de params e metricas quando a barreira e selecionada."""
    base_params = {"duration": 5, "duration_unit": "m", "contract_type": "CALL"}
    metrics = {"atr": 0.50}
    res_noop = resolve_contract_barrier_structure(base_params, metrics, "1HZ75V", TradeDirection.CALL)
    assert res_noop["contract_type"] == "CALL"
    assert "barrier" not in res_noop

    config = {
        "risk_management": {
            "barrier_contracts": {
                "enabled": True,
                "min_atr": 1.00,
                "barrier_multiplier": 0.80,
                "default_type": "ONETOUCH",
            }
        }
    }
    metrics_active = {"atr": 1.25}
    res_barrier = resolve_contract_barrier_structure(
        base_params, metrics_active, "1HZ75V", TradeDirection.CALL, config=config
    )
    assert res_barrier["contract_type"] == "ONETOUCH"
    assert res_barrier["barrier"] == "+1.00"
    assert metrics_active["barrier_contract_selected"] is True
    assert metrics_active["barrier_contract_type"] == "ONETOUCH"
