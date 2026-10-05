# Observabilidade e logs

Presentation: `app/src/presentation/terminal/logger.py`.
Contexto: `log_context.py`. SETTLE: `settle_log.py`. SSOT: `logging_config.resolve_logging_config`.
Dedupe: `log_dedupe.py`. Inventario: [`engineering-logging-inventory.md`](engineering-logging-inventory.md).

## Principios

- Logs em PT-BR, sem emoji
- Dedupe / spam filter para settlement e mensagens repetidas
- Processo > narrativa: ler `gate_reason` antes do P&L
- Logger unico do motor: `AETH` via `setup_logger` / `get_logger` (idempotente)
- Treino: `AETH.meta` / `AETH.train` — sem `print` no caminho critico
- Rich no terminal nao deve bloquear o event loop; daemon/CI: sem ANSI / logs estruturados (ver [`engineering-architecture-senior.md`](engineering-architecture-senior.md) §9)

## Knobs SSOT (`logging`)

| Chave | Default | Papel |
|-------|---------|-------|
| `level` | `INFO` | Nivel do logger AETH (`DEBUG` so para diagnostico) |
| `log_file` | `logs/engine.log` | Persistencia |
| `quiet_channels` | settle_enqueue, settle_process, settle_tolerance, settle_read, ws_ping, warmup_poll, execution_flow | Canal → DEBUG via `log_settle` |

## Prometheus e Grafana

O motor lê `telemetry.host` e `telemetry.port` de `config/settings.json` e expõe `/metrics` na porta 9100 enquanto está em execução. O bind `0.0.0.0` permite o scrape pelo Docker; o dashboard usa o datasource `http://prometheus:9090` dentro da rede Compose.

Com Docker nativo no WSL e motor Python no Windows, `make docker-up` detecta o gateway Windows pela rota padrão do WSL e o passa ao container Prometheus como `AETHER_METRICS_HOST_IP`. O alvo `host.docker.internal:9100` aponta então ao processo Windows. Para motor rodando no próprio WSL, execute `AETHER_METRICS_HOST_IP=host-gateway make docker-up`. Após reiniciar o motor e o Prometheus, confirme `up{job="aether_quantum_engine"}=1` na API do Prometheus; com o motor parado, o alvo fica DOWN por definição.

O dashboard provisionado `Aether | Visão operacional M5` apresenta apenas cartões numéricos. Todos usam consultas instantâneas do Prometheus, sem curvas, sparklines, preenchimento ou interpolação. `Coleta do motor` exibe `1` quando o scrape funciona e `0` quando falha; os demais cartões mostram `—` quando a série não existe. O exporter não emite sessão antes de conhecer saldo inicial e meta, nem radar ou contadores antes de observar o respectivo evento. `Stake da última decisão` e as probabilidades são snapshots da última decisão, não exposição atual nem resultados auditados; WIN/LOSS e EXEC/SKIP são contagens do processo atual, reiniciadas com o motor. O intervalo de atualização é 5 s, igual ao scrape.

O dashboard separado `Aether | Histórico M5` contém somente quatro séries com histórico útil: saldo, lucro, P(CALL) calibrada e edge estimado. O Grafana escolhe automaticamente a resolução conforme a largura do painel e o intervalo de tempo selecionado, respeitando o mínimo de 5 s do scrape; ambos os dashboards atualizam a cada 5 s e oferecem navegação entre si preservando o período. As séries são pontos sem linhas, preenchimento, suavização ou conexão entre lacunas. Cada consulta descarta amostras cujo último scrape tem mais de 12 s; sem amostras recentes ou com o motor desligado, o gráfico fica sem dados. Valores de inferência são snapshots repetidos pelo exporter até a próxima decisão, não novas inferências em cada scrape.

O radar de inferencia reflete a ultima compra confirmada: `aether_inference_prob` e P(CALL) bruta, `aether_inference_calibrated` e P(CALL) calibrada, e `aether_inference_payoff_edge` prioriza o EV da cotacao aceita. A margem direcional deriva da probabilidade calibrada quando nao vier explicitamente nas metricas. `aether_trading_contracts_total` usa o simbolo e o lado real da liquidacao; contratos sem lado conhecido recebem `UNKNOWN`. O Brier usa P(WIN) do lado efetivamente comprado, vinculada ao ID do contrato, e so aparece quando existe ao menos uma previsao vinculada e liquidada. `aether_trading_brier_samples` informa o tamanho da janela (ate 20). O coletor quantitativo legado nao emite seus antigos zeros quando nenhum evento o alimentou.

Na liquidacao com spots confirmados do broker, `[RESOLVED]` acrescenta `SPOT` de entrada/saida, deslocamento assinado, `REAL` (CALL, PUT ou FLAT) e a fase de entrada na vela M5. O resultado financeiro continua vindo do broker. A view `contract_label_audit` expoe `broker_spot_delta`, `m5_spot_delta`, `entry_phase_seconds` e `observed_duration_seconds` para investigar divergencias entre o contrato de 300 s e o proxy `spot_forward`; esses campos nao alteram a decisao de compra.

## Contrato de tags

| Tag | Nivel tipico | Frequencia | Consumidor |
|-----|--------------|------------|------------|
| CLUSTER / GATES / KELLY / EXEC* / RESOLVED | INFO | ≤1/ciclo (dedupe) | session-review, telemetria Prometheus |
| SETTLE.{canal} | INFO em estado; DEBUG se quiet | rate-limit canal+tick | settlement-debug |
| WSS / AUTH / MINIO | INFO no boot; DEBUG em reconexao (exceto AVISO/ERRO) | evento | deriv-connect / infra |
| RECOV (restaurado / ciclo liberado) | INFO | reconexao | cycle-debug |
| CICLO pos-liq / SRE / RECONCILE portfolio | DEBUG | rotina settle | settlement-debug |
| EXECUTION_FLOW / WARMUP | INFO se mudou; quiet → DEBUG | dedupe | cycle-debug |

Prefixo opcional de correlacao: `[cN|SYM]`.

## Tags tipicas do ciclo

| Tag | Significado |
|-----|-------------|
| `MINIO` / `TorchScript` | artefatos de modelo |
| `WSS` / `AUTH` | conexao e conta |
| `SESSAO INICIADA` | banca, stop-win |
| `DATA` / `CFG` | buffer e knobs efetivos |
| `DL` | device / inferencia |
| `CLUSTER` | Prob / Cal / Margin / Edge / raw_edge / be |
| `GATES` | LOSS_CLF FLIP/OK + skip |
| `KELLY` / `EXEC` / `RESOLVED` | sizing e resultado |
| `SIDE_EQ` / `META_VETO` | equilibrio lateral / veto meta |
| `IND` | indicadores de contexto |
| `KELLY` | p, live_wr, f*, mode |
| `EXEC` / `EXEC_EMPTY` / `EXEC_PAUSE` | ordem ou veto |
| `RESOLVED` / `RISK` | resultado e pending |
| `SETTLE` / `CICLO` / `SRE` | liquidacao e limpeza |

## Filtros

- `BlankLineSquasher` — linhas em branco consecutivas
- `SettlementSpamFilter` — SETTLE/WARMUP/EXECUTION_FLOW por **canal+tick**
- `log_dedupe` — responsabilidade de conteudo (quality/EXEC_EMPTY); Filter = anti-rajada

Nao “consertar” ausencia de trade removendo dedupe.

Diagnostico: doutrina + skill `aether-session-review`.
