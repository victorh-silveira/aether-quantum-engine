# Higienizacao do repositorio (SSOT sênior)

Runbook para morto comprovado, tooling (vulture/ruff), regra de negocio zumbi, sidecars e host WSL — sem quebrar DEMO/SSOT.

Rule: `aether-repo-hygiene.mdc`. Skill: `aether-repo-hygiene`.
Deps: [`engineering-python-deps.md`](engineering-python-deps.md). CloudOps: [`engineering-devops-cloudops-senior.md`](engineering-devops-cloudops-senior.md).
Fechamento: [`engineering-surface-sync.md`](engineering-surface-sync.md).

## Objetivo

Eliminar codigo, scripts, containers, docs, skills/rules orfaos, JSON, git/Makefile e libs sem callers — com evidencia (grep/import/compose/matriz/vulture), nao por “pouco uso”.

## 1. Codigo morto (tooling)

### Vulture (hexagonal / async)

Falsos positivos tipicos: metodos de `typing.Protocol`, hooks FastAPI lifespan, registro via `__init_subclass__`.

- Allowlist versionada: [`app/.vulture_whitelist.py`](../app/.vulture_whitelist.py)
- Pipeline: `vulture <paths> .vulture_whitelist.py --min-confidence 80` (via `clean_workspace` / pre-commit)
- Regenerar: `cd app && python -m vulture src run.py train.py scripts aether_paths.py --make-whitelist >> .vulture_whitelist.py`
- Item na allowlist que **nao** for porta Protocol / hook registrado → apagar ou refatorar (tolerancia zero a morto novo)

### Ruff (AST estrito)

Ja selecionado em [`app/pyproject.toml`](../app/pyproject.toml):

| Regra | Papel |
|-------|--------|
| F401 | import nao usado |
| F841 | local atribuido e nao lido |
| ERA001 | codigo comentado versionado (proibido; use git) |
| ARG | argumento de funcao nao usado |
| T20 / T201 | `print()` no motor — falha; scripts/`run.py`/`train.py` podem ignorar T201 |

Proibido `print()` / `rich.print` no hot path de ticks (I/O sync no loop).

## 2. Regra de negocio morta

### Feature flags

- Flag nova exige **TTL** no registro `app/src/domain/config_flag_ttl.py` (`TEMPORARY_FLAGS`) ou remocao no mesmo sprint.
- Branch `else` legado apos migracao V2 nao pode ficar meses.
- Sentinela: `test_temporary_feature_flags_not_expired` falha se flag existir apos data de expiry.
- Catalogo ativo = TCN + meta + gates; knobs SSOT permanentes em `settings.json` ficam **fora** do registro.

### Estrategias / indicadores orfaos

- Dominio so com consumidores em `application/`.
- Catalogo ativo de producao e o pipeline TCN + meta + gates (nao ha pasta `domain/strategies` paralela viva).
- Pesquisa exploratoria fora de `app/src` (ex. `research/`, barrada do motor).

### Dumps e ad-hoc

- Proibido notebooks / dumps em `app/` de producao.
- `.gitignore`: `*.parquet`, `*.arrow`, `*.pt`/`.pth`, `*.ipynb`, `research/`.

## 3. Infra (sidecars)

### Redis

TTL SSOT efemero: `REDIS_EPHEMERAL_SIG_TTL_SECONDS` = **900** (3 ciclos M5) em `app/src/infrastructure/state/redis_ephemeral_ttl.py`.

| Chave / padrao | TTL | Notas |
|----------------|-----|--------|
| `bar_sig:*` | 900 s | Dedup de epoch de barra |
| `market_sig` | 900 s | Assinatura OHLC do ciclo |
| `corr_matrix` | 900 s | Cache de correlacao |
| `settlement:queue:priority` | **sem TTL** | ZSET soberano; nunca Streams/MAXLEN |
| `session:current*` / risk / pending | **sem TTL** | Estado soberano de sessao |
| recovery/dlambert counters | **sem TTL** | Ledger operacional |

- Settlement SSOT permanece ZSET `settlement:queue:priority` — **nao** migrar para Streams; nao aplicar TTL cego que apague contratos abertos.
- Streams (MAXLEN/PEL) so se existir pipeline Streams fora do settlement (hoje: anti-padrao para liquidacao).

### Feature flags temporarias

- Registro: `app/src/domain/config_flag_ttl.py` → `TEMPORARY_FLAGS: list[tuple[str, date]]`.
- Sentinela: `test_temporary_feature_flags_not_expired` falha se `date.today() > expiry`.
- Knobs permanentes em `config/settings.json` **nao** entram no registro.
- Catalogo ativo de producao: **TCN + meta + gates** (sem pasta `domain/strategies`); pesquisa em `research/` (gitignore).

### Timescale

- Compressao ~7d; retencao ticks **30d** (ja em `004_timescale-lifecycle.sql`).
- CRAG `candle_m5` analytics; autovacuum agressivo so em tabelas high-churn de UPDATE/DELETE (nao forcar em hypertables append-only sem evidencia).

### MinIO

- Bucket SSOT `dl-models`; ILM prefixo `optuna/` ~7d (`minio-init`).
- Registry promovido separado de trials descartaveis.

## 4. Host / WSL2

Rotina periodica (dev):

```bash
# WSL
docker system prune -af --volumes   # cuidado: volumes nomeados
sudo fstrim /
bash app/scripts/wsl/clean_env.sh
```

Compactar `ext4.vhdx` no Windows apos `wsl --shutdown` (diskpart) — trimestral; **nao** automatizar diskpart no CI. O script `app/scripts/wsl/clean_env.sh` apenas lembra a rotina.

Passos manuais (Windows, Admin):

1. `wsl --shutdown`
2. `diskpart` → `select vdisk file="...\ext4.vhdx"` → `attach vdisk readonly` → `compact vdisk` → `detach vdisk`

Caches: `~/.cache/pip`, `~/.cache/torch`, `conda clean --all -y` via `clean_env.sh`.

## 5. Checklist de code review

| Vetor | Falha critica se... |
|-------|---------------------|
| Dead code | vulture/ruff vermelhos ou allowlist inchada sem Protocol |
| Business | flag legada bifurca fluxo apos TTL |
| Async | `create_task` solto sem registry/TaskGroup |
| Redis | chave transitória sem TTL; Streams no settlement |
| DB | indice/chunk sem policy; bloat sem mandato |
| I/O sync | `requests`/`pandas`/`print` no motor |

## Ondas de purge (ordem segura)

1. Morto comprovado (modulos/scripts/lixo)
2. JSON / settings sem `resolve_*`
3. Alinhamentos TF/docs
4. Deps (`aether-python-deps`)
5. QA pre-commit por onda

## Nunca remover (sem mandato)

Skills/docs na matriz; redis/timescale/minio/meta/loss; `__init__.py` de pacote; caps/settlement ZSET; historico CHANGELOG.

## Anti-padroes

Apagar Protocol “porque vulture apontou”; TTL na fila de settlement; Streams no lugar do ZSET; misturar purge com refactor hexagonal; commit unico sem QA.
