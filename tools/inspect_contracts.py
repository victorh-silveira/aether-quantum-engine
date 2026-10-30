import asyncio
import asyncpg
from datetime import datetime, timezone, timedelta

BRT = timezone(timedelta(hours=-3))

async def main():
    conn = await asyncpg.connect("postgresql://aether:aether@127.0.0.1:5432/aether")
    tables = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_name LIKE '%contract%' OR table_name LIKE '%trade%'")
    print("Tabelas encontradas:", [t['table_name'] for t in tables])
    for t in tables:
        tname = t['table_name']
        try:
            cnt = await conn.fetchval(f"SELECT count(*) FROM {tname}")
            print(f"Tabela {tname}: {cnt} registros")
            if cnt > 0:
                rows = await conn.fetch(f"SELECT * FROM {tname} LIMIT 3")
                for r in rows:
                    print(f"  {dict(r)}")
        except Exception as e:
            print(f"  Erro ao ler {tname}: {e}")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
