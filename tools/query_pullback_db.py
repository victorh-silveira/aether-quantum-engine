import asyncio
import asyncpg
from datetime import datetime, timezone, timedelta

BRT = timezone(timedelta(hours=-3))

NAMES = {
    "OTC_AEX": "Netherlands 25",
    "OTC_AS51": "Australia 200",
    "OTC_DJI": "Wall Street 30",
    "OTC_FCHI": "France 40",
    "OTC_FTSE": "UK 100",
    "OTC_GDAXI": "Germany 40",
    "OTC_HSI": "Hong Kong 50",
    "OTC_N225": "Japan 225",
    "OTC_NDX": "US Tech 100",
    "OTC_SPC": "US 500",
    "OTC_SSMI": "Swiss 20",
    "OTC_SX5E": "Euro 50",
}

async def main():
    conn = await asyncpg.connect("postgresql://aether:aether@127.0.0.1:5432/aether")
    rows = await conn.fetch("""
        SELECT symbol, period_seconds, candle_epoch, side, reason, setup_version, close
        FROM pullback_signals
        WHERE period_seconds = 900
        ORDER BY candle_epoch DESC, symbol
        LIMIT 24
    """)
    print("=" * 105)
    print("  ULTIMOS SINAIS M15 (900s) NO TIMESCALEDB (pullback_signals)")
    print("=" * 105)
    print(f"{'NOME DO ATIVO (DERIV)':<20} | {'HORARIO (BRT)':<20} | {'LADO':<8} | {'MOTIVO':<25} | {'SIMBOLO':<10}")
    print("-" * 105)
    for r in rows:
        name = NAMES.get(r["symbol"], r["symbol"])
        dt = datetime.fromtimestamp(r["candle_epoch"], tz=BRT).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{name:<20} | {dt:<20} | {r['side']:<8} | {r['reason']:<25} | {r['symbol']:<10}")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
