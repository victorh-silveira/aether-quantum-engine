# Runtime CPython 3.13 (SSOT sênior)

Documento SSOT do runtime do motor Aether no host. Complementa [`engineering-architecture-senior.md`](engineering-architecture-senior.md), [`engineering-python-deps.md`](engineering-python-deps.md) e [`AGENTS.md`](../AGENTS.md).

Rule: `aether-python-313-runtime.mdc`. Skills: `aether-python-313-runtime`, `aether-asyncio-supervisor`, `aether-polars-arrow`, `aether-torch-cuda-infer`, `aether-asyncpg-timescale`, `aether-redis-hiredis`.

## Ambiente alvo

| Item | Contrato |
|------|----------|
| Interpretador | **CPython 3.13.12** (build padrão **com GIL**) |
| Host | **WSL** Linux + Conda env `deriv-api` |
| Motor | asyncio no host; Docker só sidecars `core,ml` |
| Free-threaded (`--disable-gil`) | **Somente documentação / experimento**; **não** produção |

Build free-threaded muda ABI, locks e comportamento de C-extensions (`numpy`/`torch`/`polars`). Produção Aether permanece no build com GIL até mandato explícito + matriz de smoke ABI.

## Memória e objetos

- Todo valor Python é `PyObject*`; lifetime via **refcnt** (increment/decrement). Ciclos → GC geracional.
- **PyMalloc**: alocações ≤ **512 B** passam pelo allocator interno (pools/arenas). Acima disso → `malloc` do sistema.
- Arenas/pools: fragmentação e RSS sobem com churn de objetos pequenos no hot path (dicts temporários, strings de log, DataFrames intermediários).
- **GC Gen0–Gen2**: coleta **Stop-the-World**. Picos de alocação no loop asyncio aumentam risco de pausas no ciclo de trading.
- Preferir: reuso de buffers, structs `__slots__` no domínio quente, evitar criar grafos cíclicos grandes em application.

## Interpretador especializando (Tier 1 / Tier 2)

- **Tier 1**: specializing interpreter — caches de bytecode por tipo/atributo estável.
- **Tier 2**: uops / JIT experimental — ganho quando o caminho é monomórfico e quente.
- **Hot handlers** (WS tick, settlement poll, gate pipeline): evitar polimorfismo excessivo (muitos tipos no mesmo call site), `getattr` dinâmico em loop apertado e despacho por string em cada tick.
- Manter contratos tipados (Protocols/ports) estáveis na borda application ↔ infrastructure.

## asyncio no motor

| Conceito | Prática Aether |
|----------|----------------|
| Seletor | **epoll** no Linux/WSL |
| Filas internas | `_ready` (callbacks prontos) + `_scheduled` (timers) |
| Starvation | trabalho **síncrono > 1–2 ms** no loop atrasa WS heartbeat, settlement e TaskGroup |
| Estrutura | `asyncio.TaskGroup` + tratamento de `CancelledError` + shutdown gracioso |
| Crítico | `asyncio.shield` em writes Redis/broker que não podem morrer no meio do cancel do ciclo |
| Contexto | `ContextVars` — cópia por Task; não assume herança implícita em `to_thread` sem propagação explícita |

Offload obrigatório: PyTorch/CUDA, Polars/NumPy pesado, compressão/IO de arquivo grande → `asyncio.to_thread` / executor. Ver `predict_symbol_decision_async` → `asyncio.to_thread(eager_local_predict)`.

## Typing sênior

- **Protocols** = ports outbound na hexagonal (application depende de Protocol; infrastructure implementa).
- **ParamSpec** / `Concatenate` para wrappers genéricos (`to_thread`, retries).
- **Type parameters** (PEP 695, 3.12+): `class Repo[T]: ...` em vez de `TypeVar` legado quando o módulo já exige 3.13.
- Preferir **descriptors** e `__init_subclass__` a metaclasses pesadas.
- Domínio: tipagem estreita; sem `Any` em invariantes de risco/Kelly sem necessidade.

## Buffers e zero-copy (PEP 688)

- Buffer protocol / `memoryview` para fatias sem cópia na borda NumPy/Arrow/torch.
- Evitar `bytes(mv)` / `list(arr)` desnecessários em features e settlement payloads.
- Polars → Arrow → `to_numpy` / tensor: preferir caminho zero-copy documentado pela lib; cópia só quando layout/contiguity exige.

## Profiling e diagnóstico

| Ferramenta | Uso |
|------------|-----|
| **py-spy** / **austin** | CPU sob asyncio (não distorce o loop como cProfile sincrono) |
| **tracemalloc** | Vazamentos / picos de alocação Gen0 |
| cProfile | Offline / scripts batch; **evitar** no hot path live |

Hipótese → medir no WSL → só então mudar offload/`POLARS_MAX_THREADS`/pool sizes.

## Polars / Arrow (DataFrame SSOT)

- **Polars 1.23+**: preferir **LazyFrame**; collect consciente do event loop.
- Arrow columnar → NumPy/torch com o mínimo de cópia.
- **`pandas` é PROIBIDO** (import, `to_pandas`, dual-stack, 3ª lib DF).
- `POLARS_MAX_THREADS` não pode saturar o host a ponto de atrasar o loop/CUDA.
- DataFrame **não** é objeto de domínio: fica em application/infrastructure; domínio recebe arrays/valores tipados.

Detalhes de pin: [`engineering-python-deps.md`](engineering-python-deps.md).

## Banco, fila e inferência (bordas)

| Borda | Contrato |
|-------|----------|
| Timescale / PG | **asyncpg**: prepared statements, `COPY`/inserts em lote, pool fail-closed; gravação desacoplada da decisão |
| Redis | cliente com **hiredis**; settlement SSOT = ZSET `settlement:queue:priority` — **nunca** Streams/listas sem mandato e migração |
| Torch host | `torch.inference_mode()`, batch 1, `asyncio.to_thread`, pinned memory quando transfer GPU; `torch.compile` **opt-in offline** (não default live) |

Arquitetura completa: [`engineering-architecture-senior.md`](engineering-architecture-senior.md).

## Hexagonal e domínio puro

```
presentation → application → domain
                ↓ ports
           infrastructure
```

- **`domain/`**: sem `asyncio`, `torch`, `polars`, `httpx`, drivers DB/Redis, I/O de arquivo/rede.
- Settings: leitura via `app/settings_io.py`; parsers puros em `domain/config_knobs.py`.
- Application orquestra; infrastructure adapta.

## Anti-padrões

| Anti-padrão | Por quê | Correção |
|-------------|---------|----------|
| Sync CUDA/Polars no loop | Starvation WS/settle | `asyncio.to_thread` / executor |
| `pandas` / dual DF | Quebra SSOT e ABI | Só Polars |
| Free-threaded em prod | ABI/C-ext instável | Build com GIL |
| I/O ou torch em `domain/` | Viola hexagonal | Ports + adapters |
| DF como entidade de domínio | Acopla infra ao núcleo | VO/arrays no domínio |
| cProfile no live | Distorce asyncio | py-spy / austin |
| Redis Streams p/ settlement | Fora do SSOT | Manter ZSET `settlement:queue:priority` |

CloudOps detalhado: [`engineering-devops-cloudops-senior.md`](engineering-devops-cloudops-senior.md).
| `torch.compile` default live | Risco de warmup/latência | Opt-in offline |
| Polimorfismo em handler quente | Quebra specializing | Call sites monomórficos |
| Sync >2 ms sem offload | Atrasa `_ready` | Medir e offload |
| Metaclasse pesada | Complexidade / tipagem | `__init_subclass__` / descriptors |
| Ignorar GC STW | Pausas no ciclo | Menos churn / reuso de buffers |

## Referências cruzadas

- [`engineering-architecture-senior.md`](engineering-architecture-senior.md) — host, DDD, sidecars, event loop
- [`engineering-python-deps.md`](engineering-python-deps.md) — pins, Polars-only, ABI
- [`AGENTS.md`](../AGENTS.md) — entrada de agentes, universo ops, proibições
- Skills Cursor: `.cursor/skills/aether-python-313-runtime/`, `aether-asyncio-supervisor/`, `aether-polars-arrow/`, `aether-torch-cuda-infer/`, `aether-asyncpg-timescale/`, `aether-redis-hiredis/`
