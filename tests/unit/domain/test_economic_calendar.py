"""Testes unitarios (100% cobertura) do gate de calendario economico e cap regional."""

from __future__ import annotations

from src.domain.fundamental import (
    CalendarPolicy,
    EconomicEvent,
    apply_regional_block_cap,
    evaluate_calendar_veto,
    event_affects_symbol,
    policy_from_config,
    symbol_regional_block,
)


def test_symbol_regional_block_and_policy_from_config():
    assert symbol_regional_block("OTC_GDAXI") == "EUROPE"
    assert symbol_regional_block("OTC_SPC") == "US"
    assert symbol_regional_block("OTC_N225") == "ASIA_PAC"
    assert symbol_regional_block("R_100") == "R_100"

    default_policy = policy_from_config({})
    assert default_policy.enabled is False
    assert default_policy.impacts == frozenset({"HIGH"})
    assert default_policy.pre_minutes == 15
    assert default_policy.post_minutes == 0

    custom = policy_from_config(
        {
            "economic_calendar": {
                "enabled": True,
                "impacts": ["High", "Medium"],
                "usd_medium_enabled": True,
                "usd_global": False,
                "pre_minutes": 10,
                "post_minutes": 5,
                "fail_open": False,
            }
        }
    )
    assert custom.enabled is True
    assert custom.impacts == frozenset({"HIGH", "MEDIUM"})
    assert custom.usd_global is False
    assert custom.pre_minutes == 10
    assert custom.post_minutes == 5
    assert custom.fail_open is False


def test_event_affects_symbol_and_veto_windows():
    ev_usd_high = EconomicEvent("Core PCE", "USD", 1_795_613_400, "High")
    ev_usd_med = EconomicEvent("CB Confidence", "USD", 1_795_613_400, "Medium")
    ev_eur_high = EconomicEvent("ECB Rate", "EUR", 1_795_613_400, "High")
    ev_jpy_high = EconomicEvent("BOJ Rate", "JPY", 1_795_613_400, "High")

    policy = CalendarPolicy(enabled=True, impacts=frozenset({"HIGH"}), usd_global=True, pre_minutes=15, post_minutes=0)
    assert event_affects_symbol(ev_usd_high, "OTC_GDAXI", policy) is True
    assert event_affects_symbol(ev_usd_med, "OTC_GDAXI", policy) is False
    assert event_affects_symbol(ev_eur_high, "OTC_GDAXI", policy) is True
    assert event_affects_symbol(ev_eur_high, "OTC_SPC", policy) is False
    assert event_affects_symbol(ev_jpy_high, "OTC_N225", policy) is True

    policy_usd_med = CalendarPolicy(
        enabled=True, impacts=frozenset({"HIGH"}), usd_medium_enabled=True, usd_global=False
    )
    assert event_affects_symbol(ev_usd_med, "OTC_SPC", policy_usd_med) is True
    assert event_affects_symbol(ev_usd_med, "OTC_GDAXI", policy_usd_med) is False

    # Desabilitado ou periodo invalido -> nao veta
    assert evaluate_calendar_veto("OTC_SPC", 1_795_613_400, 900, [ev_usd_high], CalendarPolicy(enabled=False)) == (
        False,
        "",
    )
    assert evaluate_calendar_veto("OTC_SPC", 1_795_613_400, 0, [ev_usd_high], policy) == (False, "")

    # Pre-release 15m antes (entry 13:15, news 13:30) -> VETA
    vetoed, reason = evaluate_calendar_veto("OTC_GDAXI", 1_795_613_400 - 900, 900, [ev_eur_high], policy)
    assert vetoed is True
    assert reason == "veto_calendario_macro:EUR:HIGH"

    # Pos-release +15m (entry 13:45 apos news 13:30, post_minutes=0) -> LIVRE (preserva fade pos-choque)
    assert evaluate_calendar_veto("OTC_GDAXI", 1_795_613_400 + 900, 900, [ev_usd_high], policy) == (False, "")

    # Evento de outro bloco nao veta
    assert evaluate_calendar_veto("OTC_SPC", 1_795_613_400 - 900, 900, [ev_eur_high], policy) == (False, "")


def test_apply_regional_block_cap():
    orders = [
        ("OTC_GDAXI", "CALL", {}),
        ("OTC_FCHI", "CALL", {}),
        ("OTC_FTSE", "CALL", {}),
        ("OTC_SPC", "CALL", {}),
        ("OTC_NDX", "CALL", {}),
        ("OTC_N225", "CALL", {}),
    ]
    assert apply_regional_block_cap(orders, 0) == orders
    capped = apply_regional_block_cap(orders, 2)
    assert [o[0] for o in capped] == ["OTC_GDAXI", "OTC_FCHI", "OTC_SPC", "OTC_NDX", "OTC_N225"]
