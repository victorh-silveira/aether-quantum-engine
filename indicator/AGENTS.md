# AGENTS.md — indicador público Aether

## Contrato vigente

O serviço lê apenas o catálogo público e `ticks_history` da Deriv. Calcula o setup fixo `senior_institutional` com velas OHLC fechadas por ativo e período M15/H1/D1. Somente M15 da versão em `config/settings.json` pode gerar proposta no motor autenticado. O indicador não acessa conta, não compra contratos e não calcula stake.

A regra operacional exige tendência EMA20/EMA50, recuo contrário até 0,5 ATR14, confirmação, referência de preço típico da sessão UTC com bandas, FVG efetivo e killzone BRT. A referência usa OHLC sem volume e não é VWAP. Para OTC, intervalos entre sessões UTC de até 72 horas são preservados sem interpolação; lacunas intradiárias e vela de confirmação atrasada produzem `SEM SINAL`.

`make app-train` mede apenas o proxy direcional de velas e publica ranking diagnóstico. Não concatena cotações Yahoo com as velas da Deriv, não altera settings, não muda a âncora e não promove contratos. Versões novas entram como `piloto`. O registro muda para `ativo` apenas após 200 compras DEMO únicas e liquidadas pelo broker na mesma versão, ativo e período, com limite inferior unilateral de 95% do retorno líquido médio positivo. REAL ainda depende do EV da proposta final no motor.

O treino imprime entradas M15 por ativo no dia BRT, na ultima data com velas e em cada data historica com setup. O backtest usa a direcao da vela seguinte e payout hipotetico com stake fixa de uma unidade; nao representa compras nem liquidacoes do broker. `s/d` em HOJE indica que nao ha vela fechada do ativo na data atual.

## Engenharia

- Responder, documentar e nomear commits em PT-BR; trabalhar no WSL com Python 3.13.
- Preservar domínio puro, I/O nos adaptadores e séries sem interpolação.
- Exibir `SEM SINAL` e motivo quando faltarem dados, vela atual, padrão ou versão; nunca reutilizar lado antigo.
- Distinguir proxy histórico, sinal observado, piloto DEMO e liquidação confirmada pelo broker.
- Atualizar código, testes, dashboards, documentação, rules e skills quando mudar a semântica.
- Executar `make app-lint` e `make app-test` na raiz. A cobertura em `app/src` e `indicator/app/src` deve permanecer em 100%.
- O Compose e o Grafana são compartilhados com o motor. `make docker-up` ou `make app-train` inicializam a infraestrutura; `make app-run` é o processo autenticado separado.

Leia [estratégia](../docs/strategy-pullback.md), [arquitetura](../docs/arquitetura.md) e [operação](docs/operacao.md).
