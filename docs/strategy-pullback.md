# Estrategia de tendencia, liquidez e recuo institucional

## 1. Sinal e Matriz de Confluencia M5

O subsistema publico consulta `active_symbols` e `ticks_history` para os 12 indices sinteticos de acoes OTC (replicas dos ETFs mundiais SPY, QQQ, DIA, EWG, FEZ, etc.) e indices de volatilidade da Deriv. O treino observa M5, M15 e H1 de forma independente; somente sinais M5 podem abrir contratos Rise/Fall de 5 minutos. 

A variante ativa configurada no SSOT e a **`senior_institutional`** (setup institucional senior), que avalia 128 velas fechadas e continuas para aquecimento da EMA50 e exige confluencia estrita de 5 pilares:

### Pilar 1: Tendencia Estrutural
- **CALL**: EMA20 acima da EMA50 (`fast[-1] > slow[-1]`) com EMA20 estritamente crescente nas tres velas anteriores (`fast[-1] > fast[-2] > fast[-3] > fast[-4]`) e inclinacao positiva da EMA50 (`slow[-2] > slow[-5]`).
- **PUT**: Espelhamento exato com EMA20 abaixo da EMA50 e ambas decrescentes.

### Pilar 2: Retracao e Volatilidade (Pullback)
- A vela anterior (recuo) deve penetrar a no maximo 0,5 ATR14 da EMA20 (`pullback.low <= fast[-2] + 0.5 * atr` no CALL) e fechar acima da EMA50.
- A vela de recuo deve ser contraria a tendencia (`pullback.close < pullback.open` no CALL).
- A vela de confirmacao fecha acima da maxima do recuo com corpo entre 0,03 e 1,5 ATR14 e distancia da EMA20 restrita a 0,75 ATR14.

### Pilar 3: Session VWAP e Bandas de Dispersao (+/- 1.5 sigma)
- Calculo da VWAP Intraday ancorada no inicio da sessao diaria (00:00 UTC):
  $$VWAP = \frac{\sum P_{typical} \times Volume}{\sum Volume}$$
  onde $P_{typical} = \frac{High + Low + Close}{3}$.
- Desvio padrao da sessao:
  $$\sigma = \sqrt{\frac{\sum Volume \times (P_{typical} - VWAP)^2}{\sum Volume}}$$
- **Filtro de Desvio**:
  - No CALL, o fechamento deve estar acima ou igual a VWAP e contido na banda superior de $+1.5\sigma$ ($Close \le VWAP + 1.5\sigma$). Previne compras no topo de exaustao de volatilidade.
  - No PUT, o fechamento deve estar abaixo ou igual a VWAP e contido na banda inferior de $-1.5\sigma$ ($Close \ge VWAP - 1.5\sigma$). Previne vendas no fundo de exaustao.

### Pilar 4: Mitigacao de Fair Value Gap (FVG)
- Identifica desequilíbrio institucional (*imbalance*) entre as velas $t-2$ e $t$:
  - Bullish FVG: `low[t] > high[t-2]`
  - Bearish FVG: `high[t] < low[t-2]`
- Valida que o recuo testou e mitigou a regiao de desequilibrio antes do rompimento, eliminando entradas em falso rompimento.

### Pilar 5: Hard Gate de Killzones (Horario de Brasilia - BRT / UTC-3)
- O timing de execucao obedece aos maiores picos de liquidez e volume financeiro global:
  - **London Open**: 03:00 as 08:00 BRT
  - **New York Open / Overlap**: 10:00 as 13:00 BRT
  - **London Fix / Close**: 14:00 as 16:00 BRT
- Sinais gerados fora dessas janelas horarias recebem `SEM SINAL` com motivo `Fora da killzone institucional de liquidez`.

---

## 2. Diagnostico, Aprendizado Horario e Promocao

O comando `make app-train` sobe a infraestrutura de dados e executa o pipeline de diagnostico e aprendizado:
1. Coleta historico continuo M5, M15 e H1 para os 12 ativos mundiais OTC e indices de volatilidade.
2. Integra cotacoes reais de mercado via API HTTP do Yahoo Finance v8 (`real_market_feed.py`), unindo a serie sintetica com os precos dos ativos subjacentes.
3. Avalia o desempenho pelo proxy de velas (abertura ate fechamento da vela seguinte) em janelas cronologicas disjuntas.
4. **Calculo do Perfil Horario BRT**: Analisa a distribuicao de acertos hora a hora no fuso de Brasilia (UTC-3), identificando as horas com taxa de acerto superior a 50% (`promoted_optimal_hours_brt`) e a janela continua ideal de negociacao (`promoted_best_window_brt`).
5. **Ranqueamento Soberano por EV**:
   $$EV = p \times Payout - (1 - p)$$
   com payout base de 85% ($0.85$). O ativo com maior EV positivo e menor sequencia de perdas e promovido automaticamente para `config/settings.json` e `drift_symbols.py`. O ativo promovido vigente e **OTC_SX5E** (Euro Stoxx 50 / FEZ).

---

## 3. Execucao, Risco e Liquidador

- **Conta DEMO**: Avalia ate 200 compras unicas por versao com stake base de 1% da banca disponivel. Apenas contratos liquidados com `settlement_source='broker'` sao computados.
- **Conta REAL**: Exige limite inferior unilateral de 95% do retorno medio do piloto DEMO positivo, probabilidade calibrada via posterior Beta(0,5; 0,5), EV positivo e margem de 0,01 acima do break-even na cotacao final do broker.
- **Recuperacao Institucional**: Amortizacao linear 1/1 baseada no payout liquido cotado, com **teto maximo de 5% por ordem**. Vetados absolutamente Kelly, Martingale ou D'Alembert geometricos.
- **Protecao e Quarentena**: Cada ordem reserva conta/ativo/vela M5 via `SET NX` no Redis. Reconciliacao automatica deduplica por `contract_id` e quarentena de 900 segundos e ativada em timeouts sem confirmacao de compra.
