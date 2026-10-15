---
name: aether-indicator
description: Desenvolve e revisa o indicador publico de indices Volatility, treino temporal, sinais sem contrato e dashboards Grafana.
---

# Indicador Aether

1. Leia `AGENTS.md`, `docs/arquitetura.md`, `docs/modelos.md` e `docs/operacao.md`.
2. Restrinja treino, serviço e dashboards aos Volatility `R_<número>` e `1HZ<número>V` ativos; registre progresso e prazo por ativo. Preserve a barreira do cliente público: apenas catálogo e candles; teste contra rotas de negociação.
3. Ao alterar sinais, teste fechamento de vela, lacunas, isolamento de ativo/período, modelo ausente e frescor.
   Separe a previsão confirmada da observação provisória de preço em formação; atualize esta a cada 30 segundos sem reescrever a previsão emitida.
4. Ao alterar treino, compare validação temporal com referência simples, registre as métricas dos pares experimentais e arquive checkpoints aprovados quando forem reclassificados. Nunca apresente leitura experimental como vantagem comprovada.
   Separe selecao, calibracao e validacao em blocos cronologicos. Registre as candidatas acima do break-even de referencia sem tratar closes M5 como resultado de contrato ou promocao para REAL.
5. Atualize Grafana, Prometheus, docs e regra junto com o código. As visões de sinais mostram apenas a última leitura CALL/PUT atual e com versão de modelo ativa; exigem modelo `APROVADO`; o catálogo separado lista todos os índices e mostra `EXPERIMENTAL` como `SEM SINAL`. Exiba fechamento real em `candle_epoch + período` e horizonte em `candle_epoch + 2 * período`. Indicador e motor compartilham Grafana em `localhost:3000` com `admin`/`aether`.
   Teste a expansão das variáveis do Grafana também quando não houver sinal e os seletores estiverem vazios.
6. Preserve `make help` e os grupos `app-*`/`docker-*` da `main`. Na raiz, `make docker-up` sobe a stack unica e `make app-train` treina indicador M5/M15/H1, loss, TCN e meta; o meta usa os mesmos indices M5 selecionados para a TCN. Rode `make app-lint` e `make app-test` na raiz para validar motor e indicador, além do Compose e dos dashboards. Não afrouxe a cobertura de 100%.
7. Preserve a estrutura de jobs do CI anterior nesta branch e seus gatilhos próprios. Na integração, o CI da `main` valida indicador e motor antes de publicar; o indicador permanece sem rota de negociação.
