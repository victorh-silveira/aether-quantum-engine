from src.application.services.llm import llm_bridge as bridge
from src.application.services.llm.llm_repeat_guard import can_repeat_same_direction
from src.domain.models.trade import TradeDirection


def test_can_repeat_same_direction_requires_neutral_rsi_m3_and_wick():
    runtime = {
        "same_direction_strict_enabled": True,
        "same_direction_rsi_min": 40,
        "same_direction_rsi_max": 60,
        "same_direction_require_m3_confirmation": True,
        "same_direction_require_wick_confirmation": True,
    }
    assert (
        can_repeat_same_direction(
            TradeDirection.CALL,
            "nota com pavio de rejeicao",
            "M3 gatilho: RSI=50.0 zona=neutro; vela=bullish; tendencia_EMA=alta;",
            "Normal",
            runtime,
        )
        is True
    )


def test_can_repeat_same_direction_neutral_sem_wick_falha_quando_exige():
    runtime = {
        "same_direction_strict_enabled": True,
        "same_direction_rsi_min": 40,
        "same_direction_rsi_max": 60,
        "same_direction_require_m3_confirmation": True,
        "same_direction_require_wick_confirmation": True,
    }
    assert (
        can_repeat_same_direction(
            TradeDirection.CALL,
            "nota neutra curta sem tokens",
            "M3 gatilho: RSI=50.0 zona=neutro; vela=bullish; tendencia_EMA=alta;",
            "Normal",
            runtime,
        )
        is False
    )


def test_can_repeat_same_direction_wait_retorna_false():
    assert (
        can_repeat_same_direction(
            None,
            "qualquer",
            "M3 lateral",
            "Normal",
            {"same_direction_strict_enabled": False},
        )
        is False
    )


def test_can_repeat_same_direction_sem_strict_permite():
    runtime = {"same_direction_strict_enabled": False}
    assert (
        can_repeat_same_direction(
            TradeDirection.PUT,
            "qualquer",
            "M3 gatilho: RSI=40 zona=neutro; vela=bearish;",
            "Normal",
            runtime,
        )
        is True
    )


def test_can_repeat_same_direction_strict_sem_confirmacao_m3_falha():
    runtime = {
        "same_direction_strict_enabled": True,
        "same_direction_require_m3_confirmation": True,
        "same_direction_require_wick_confirmation": False,
    }
    assert (
        can_repeat_same_direction(
            TradeDirection.CALL,
            "wick ok",
            "M3 gatilho: RSI=55 zona=neutro; vela=bearish;",
            "Normal",
            runtime,
        )
        is False
    )


def test_can_repeat_same_direction_strict_desliga_confirmacao_m3():
    runtime = {
        "same_direction_strict_enabled": True,
        "same_direction_require_m3_confirmation": False,
        "same_direction_require_wick_confirmation": False,
    }
    assert (
        can_repeat_same_direction(
            TradeDirection.CALL,
            "sem wick",
            "M3 gatilho: RSI=n/d zona=neutro; vela=bullish; dist_EMA21=0.10%.",
            "Normal",
            runtime,
        )
        is True
    )


def test_choose_direction_without_wait_so_propaga_call_put():
    assert (
        bridge._choose_direction_without_wait(
            TradeDirection.CALL,
            "M15 tendencia_EMA=alta",
            "M5 tendencia_EMA=baixa",
            "M3 gatilho tendencia_EMA=baixa",
            TradeDirection.PUT,
        )
        == TradeDirection.CALL
    )
