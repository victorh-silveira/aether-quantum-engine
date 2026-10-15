# Indicador público de índices Volatility

O serviço lê `active_symbols` e `ticks_history` sem credenciais de conta. Emite CALL, PUT ou SEM SINAL para cada índice Volatility e período M5/M15/H1, com a versão fixa do [setup de tendência e recuo](../docs/strategy-pullback.md). Usa apenas velas fechadas e contíguas; a vela em formação não entra no sinal. O serviço não propõe nem compra contratos, e os sinais sintéticos não são previsões de Forex.

Na raiz do projeto, execute:

```bash
make docker-up
make app-train
make app-run
```

`make app-train` coleta 5.000 velas contíguas por índice/período e imprime frequência e acerto direcional como **proxy de velas**. O diagnóstico não promove versões. O motor em DEMO testa até 200 setups com stake mínima; apenas contratos únicos liquidados pelo broker podem habilitar REAL. `make app-research-ml` mantém TCN/meta/loss como pesquisa separada.

O Grafana em `http://localhost:3000` mostra setups atuais, estado do piloto, cotações e resultados confirmados. O painel de catálogo inclui também índices sem sinal. Não há interpolação de lacunas nem probabilidade fabricada para um setup DEMO. Execute `make app-lint` e `make app-test` para validar motor e indicador com cobertura de 100%.
