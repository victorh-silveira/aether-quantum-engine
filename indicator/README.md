# Indicador público de índices de ações mundiais (OTC) e Volatility

O serviço lê `active_symbols` e `ticks_history` da Deriv sem credenciais de negociação e integra cotações reais de mercado via API HTTP pública do Yahoo Finance v8 para os 12 ativos de ações mundiais (réplicas dos ETFs SPY, QQQ, DIA, EWG, FEZ, etc.).

Emite `CALL`, `PUT` ou `SEM SINAL` para cada ativo e período M5/M15/H1, utilizando a variante **`senior_institutional`** do [setup institucional de tendência, liquidez e recuo](../docs/strategy-pullback.md). A avaliação utiliza 128 velas fechadas e contíguas com validação estrita de Session VWAP ($\pm 1.5\sigma$), mitigação de Fair Value Gap (FVG) e Hard Gate de Killzones no Horário de Brasília (BRT / UTC-3). Velas em formação não entram no cálculo de decisão.

Na raiz do projeto, execute:

```bash
make docker-up
make app-train
make app-run
```

- `make app-train`: Coleta séries contínuas por ativo/período, diagnostica frequência e acerto pelo proxy de velas, calcula o perfil de assertividade por hora no fuso BRT e ranqueia os pares por expectativa matemática (EV esperado no payout 85%). O par promovido para execução direta é o **`OTC_SX5E`** (Euro Stoxx 50 / FEZ).
- `make app-research-ml`: Mantém os experimentos de TCN, Meta-Learner e Loss-Classifier como pesquisa analítica isolada.
- O Grafana em `http://localhost:3000` exibe setups em tempo real, estado dos pilotos e cotações confirmadas.

Execute `make app-lint` e `make app-test` para certificar 100% de cobertura de testes unitários.
