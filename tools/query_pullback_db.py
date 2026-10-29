import asyncio
import asyncpg
from datetime import datetime, timezone, timedelta

BRT = timezone(timedelta(hours=-3))

async def main():
    conn = await asyncpg.connect("postgresql://aether:aether@127.0.0.1:5432/aether")
    rows = await conn.fetch("""
        SELECT symbol, period_seconds, candle_epoch, side, reason, setup_version, close
        FROM pullback_signals
        WHERE period_seconds = 900
        ORDER BY candle_epoch DESC, symbol
        LIMIT 24
    """)
    print("=" * 95)
    print("  ULTIMOS SINAIS M15 (900s) NO TIMESCALEDB (pullback_signals)")
    print("=" * 95)
    print(f"{'SIMBOLO':<12} | {'HORARIO VELA (BRT)':<20} | {'LADO':<8} | {'MOTIVO':<25} | {'VERSAO':<22}")
    print("-" * 95)
    for r in rows:
        dt = datetime.fromtimestamp(r["candle_epoch"], tz=BRT).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{r['symbol']:<12} | {dt:<20} | {r['side']:<8} | {r['reason']:<25} | {r['setup_version']:<22}")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
