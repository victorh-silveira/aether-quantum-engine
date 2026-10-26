# Indicador público de índices OTC e Volatility

O serviço consulta `active_symbols` e `ticks_history` da Deriv sem credenciais de negociação. Usa apenas velas OHLC fechadas da Deriv para calcular o setup `senior_institutional` em M15, H1 e D1. Velas externas não completam nem substituem a série do contrato.

O [setup institucional M15](../docs/strategy-pullback.md) exige tendência EMA20/EMA50, recuo contrário, referência OHLC da sessão UTC, FVG efetivo e killzone BRT. A referência da sessão usa preços típicos por vela, sem volume; não é VWAP. Somente M15 da versão operacional configurada pode chegar ao motor de contratos. O cliente público nunca compra.

Na raiz:

```bash
make docker-up
make app-train
make app-run
```

`make app-train` apresenta frequência, acerto direcional por proxy de velas, perfil horário e ranking diagnóstico. Não troca a âncora `OTC_SSMI`, não marca uma versão como ativa e não prova vantagem econômica. A promoção para REAL depende de 200 liquidações DEMO confirmadas pelo broker na mesma versão, ativo e período, além do limite inferior de retorno positivo. H1 e D1 são pesquisa.

O Grafana em `http://localhost:3000` separa observações, pilotos e contratos confirmados. Use `make app-lint` e `make app-test` para validar o código e a cobertura.
