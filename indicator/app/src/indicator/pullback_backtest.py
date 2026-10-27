"""Apresentacao do backtest de velas M15 por ativo e dia BRT."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

BRT = timezone(timedelta(hours=-3))


def _net(wins: int, losses: int, payout: float) -> str:
    return f"{wins * payout - losses:+.2f}"


def format_daily_backtest(report: dict[str, dict], now: int, payout: float) -> str:
    """Mostra setups historicos por dia sem simular liquidacao de contrato."""
    today = datetime.fromtimestamp(now, tz=BRT).date().isoformat()
    rows = [(key.split(":")[0], row) for key, row in sorted(report.items()) if key.endswith(":900")]
    lines = [
        f"BACKTEST M15 POR ATIVO | hoje={today} BRT | payout hipotetico={payout:.2%}",
        "Entradas = setups historicos; resultado = direcao da proxima vela OHLC da Deriv.",
        "ATIVO     | ULTIMA VELA | HOJE | ULT. DIA | TOTAL | W/L/E   | SALDO/U | ESTADO",
    ]
    for symbol, row in rows:
        state = row.get("estado", "SEM DADOS")
        if state != "PROXY DE VELAS":
            lines.append(
                f"{symbol:<9} | {'n/a':<11} | {'n/a':<4} | {'n/a':<8} | {'n/a':<5} | {'n/a':<7} | {'n/a':<7} | {state}"
            )
            continue
        daily = row.get("por_dia_brt", {})
        latest = row.get("ultima_vela_brt", "n/a")
        today_count = str(daily.get(today, {}).get("setups", 0)) if latest == today else "s/d"
        latest_count = daily.get(latest, {}).get("setups", 0)
        wins, losses, ties = (int(row.get(key, 0)) for key in ("acertos", "erros", "empates"))
        lines.append(
            f"{symbol:<9} | {latest:<11} | {today_count:<4} | {latest_count:<8} | "
            f"{row.get('setups', 0):<5} | {wins}/{losses}/{ties:<3} | {_net(wins, losses, payout):<7} | {state}"
        )
    lines.extend(
        [
            "s/d = sem vela fechada hoje; ULT. DIA = entradas na data da ultima vela disponivel.",
            "SALDO/U = soma hipotetica com stake fixa de 1 unidade; empate vale 0 nesta simulacao.",
            "DETALHE DOS DIAS COM ENTRADAS (BRT):",
            "ATIVO     | DIA        | ENTRADAS | W/L/E   | SALDO/U",
        ]
    )
    any_daily = False
    for symbol, row in rows:
        for day, stats in sorted(row.get("por_dia_brt", {}).items()):
            any_daily = True
            wins, losses, ties = (int(stats.get(key, 0)) for key in ("acertos", "erros", "empates"))
            lines.append(
                f"{symbol:<9} | {day} | {stats['setups']:<8} | {wins}/{losses}/{ties:<3} | {_net(wins, losses, payout)}"
            )
    if not any_daily:
        lines.append("Nenhum setup M15 no historico consultado.")
    lines.append("Proxy offline: sem cotacao, horario real de compra, custos ou liquidacao broker; nao qualifica REAL.")
    return "\n".join(lines)
