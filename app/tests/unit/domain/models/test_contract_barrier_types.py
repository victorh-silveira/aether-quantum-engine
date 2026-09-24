"""Testes unitarios para modelos de dominio de contratos de barreira."""

from src.domain.models.contract_barrier_types import (
    BarrierContractConfig,
    ContractFamily,
    ResolvedBarrierContract,
)


def test_contract_family_enum():
    """Verifica membros da enumeracao de familias de contratos."""
    assert ContractFamily.RISE_FALL.value == "RISE_FALL"
    assert ContractFamily.TOUCH.value == "TOUCH"
    assert ContractFamily.NO_TOUCH.value == "NO_TOUCH"


def test_barrier_contract_config_defaults():
    """Verifica instanciacao e defaults de BarrierContractConfig."""
    cfg = BarrierContractConfig()
    assert cfg.enabled is False
    assert cfg.min_atr == 1.20
    assert "explosion" in cfg.target_regimes
    assert cfg.barrier_multiplier == 0.80
    assert cfg.default_type == "ONETOUCH"


def test_resolved_barrier_contract_instantiation():
    """Verifica instanciacao de ResolvedBarrierContract."""
    resolved = ResolvedBarrierContract(
        contract_type="ONETOUCH",
        barrier="+1.50",
        duration=300,
        duration_unit="s",
        is_barrier=True,
    )
    assert resolved.contract_type == "ONETOUCH"
    assert resolved.barrier == "+1.50"
    assert resolved.is_barrier is True
