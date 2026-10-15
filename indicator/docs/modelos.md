# Modelos e limites dos sinais

## Setup operacional atual

O sinal em producao e o setup fixo `pullback-1`, documentado em [`docs/strategy-pullback.md`](../../../docs/strategy-pullback.md). Cada par ativo/período usa sua propria versao e somente velas fechadas contiguas. O `make app-train` prepara dados e reporta acertos entre a abertura e o fechamento da vela seguinte como **proxy de velas**; nao ajusta parametros nem aprova contratos. As secoes de pesquisa de ML abaixo descrevem artefatos historicos sem autorizacao operacional.

Cada ativo e período M5/M15/H1 tem modelo independente. O treino exige 5.000 velas fechadas, contínuas e atuais consultadas na granularidade própria da Deriv. Lacunas e épocas conflitantes não são interpoladas; faixas faltantes recebem até três tentativas de reconsulta. Onde M5 e M15/H1 se sobrepõem, as barras diretas são comparadas à agregação completa de M5.

A pesquisa compara referência majoritária, momentum, regressão logística em retornos, regressão com features OHLC causais e uma árvore rasa. São cinco configurações por par e período. Três janelas walk-forward anteriores escolhem a configuração com embargo do lookback e horizonte. Ajuste, calibração e teste respeitam ordem temporal; os 20% finais não participam da escolha. A aprovação estatística exige Brier melhor que a melhor referência, limite inferior de 95% da melhoria acima de zero por bootstrap em blocos e ECE até 0,05. Probabilidades não são ampliadas para fabricar EV.

O relatório mostra acurácia, Brier, ECE, distribuição de probabilidades, candidatas acima do break-even de referência e resultado por regime. `OK` significa apenas aprovação estatística. O registro de versões separa `pesquisa`, `piloto`, `ativo` e `reprovado` por ativo e período. Novos ajustes de uma versão em piloto ou ativa são guardados em `data/indicator-models/candidates/<versao>/`; suas previsões são gravadas em `indicator_candidate_signals`, sem reescrever o histórico do modelo principal.

A coluna `correct` da leitura principal compara direção prevista e próximo fechamento do mesmo sintético. Ela não representa liquidação de contrato. Promoção para REAL depende de 200 contratos DEMO confirmatórios da mesma versão, ativo e período, liquidados pelo broker, com limite inferior unilateral de 95% do retorno líquido médio acima de zero. Compras exploratórias não contam. Cada previsão futura ainda pode falhar; o indicador não prevê pares Forex.
