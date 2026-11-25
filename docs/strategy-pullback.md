# Estratégia Quantitativa: Pullback Institucional M15

## 1. Visão Geral da Estratégia

O Aether Quantum Engine adota como variante ativa o **Persistent Trend v3 (`persistent_trend_v3`, versão `pullback-1-876b50614185`)** operando contratos Rise/Fall de 15 minutos (900 segundos) na corretora Deriv, preservando `multi_regime_hf`, `dual_sniper` e `senior_institutional` para retrocompatibilidade e rollback.

O universo operacional abrange os **12 ativos mundiais de ações OTC** (âncora `OTC_FTSE`; Europa: `OTC_GDAXI`, `OTC_FCHI`, `OTC_FTSE`, `OTC_SX5E`, `OTC_AEX`, `OTC_SSMI`; EUA: `OTC_SPC`, `OTC_NDX`, `OTC_DJI`; Ásia-Pacífico: `OTC_N225`, `OTC_AS51`, `OTC_HSI`), com análise 100% causal em velas M15/H1 fechadas e execução multi-ordem controlada (`multi_symbol_orders=true`, `max_orders_per_cycle=4`, `cycle_risk_budget_pct=0.04`).

### Ramos Causais do `persistent_trend_v3` (Validados em H1 e H2 de 2026)
1. **Ramo B1 (`h1_di24_40`):** alinhamento `H1 EMA20/50` com `DI(24) > +40` (CALL) ou `< -40` (PUT).
2. **Ramo B2 (`h1_di14_45`):** alinhamento `H1 EMA20/50` com `DI(14) > +45` (CALL) ou `< -45` (PUT).
3. **Ramo B3 (`di14_pb45`):** `|DI(14)| > 45` com recuo de 1 vela contra a tendência (pullback).
4. **Ramo B4 (`di28_cont38`):** `|DI(28)| > 38` com continuação de 1 vela a favor da tendência.
5. **Ramo B5 (`di20_adx35_40`):** força estrutural `ADX(20) >= 35` combinada com `|DI(20)| > 40`.
- **Guardas de Qualidade & Resolução de Conflitos:** exclusão de horas estruturalmente tóxicas (`03h, 06h, 19h UTC`), `veto_conflito` se dois ramos dispararem lados opostos no mesmo `(symbol, epoch)` e contrato canônico `reason="setup_confirmado"`.
- **Gate de Calendário Econômico (`economic_calendar`):** veto cirúrgico pré-release e overlap de contrato (`pre_minutes=15`, `post_minutes=0`, impactos `High`, `USD` global e demais moedas por bloco regional) via feed semanal com cache local de 6h e `fail_open=true`.

---

## 2. A Matriz dos 5 Pilares Institucionais

Uma oportunidade de compra (CALL ou PUT) só é validada quando atende **concomitantemente** aos 5 pilares estruturais sobre velas M15 perfeitamente fechadas:

### Pilar 1: Tendência Estrutural
- **CALL:** A média móvel exponencial de 20 períodos ($	ext{EMA}_{20}$) deve estar estritamente acima da média de 50 períodos ($	ext{EMA}_{50}$), com inclinação positiva nas últimas 3 transições.
- **PUT:** $	ext{EMA}_{20} < 	ext{EMA}_{50}$ com inclinação descendente.

### Pilar 2: Retração Técnica de Volatilidade
- A vela anterior à confirmação deve ter realizado um recuo em direção à média.
- A mínima da vela de recuo (para CALL) deve estar contida em até **$0,5 	imes 	ext{ATR}_{14}$** da $	ext{EMA}_{20}$, sem fechar abaixo da $	ext{EMA}_{50}$.
- A vela de confirmação subsequente deve fechar a favor da tendência com corpo entre $0,03$ e $1,5 	imes 	ext{ATR}_{14}$.

### Pilar 3: Referência OHLC da Sessão UTC
- Cálculo do preço típico acumulado da sessão UTC iniciado às 00:00 UTC:
  $$	ext{Preço Típico} = rac{	ext{High} + 	ext{Low} + 	ext{Close}}{3}$$
- **Bandas de Dispersão:** Bandas de tolerância de $\pm 1,5 \sigma$ (desvio padrão) em torno da média de preços típicos.
- O fechamento da vela de confirmação deve ser favorável à referência e permanecer contido dentro das bandas de $\pm 1,5 \sigma$ (evitando exaustão estatística).
- *Nota de Engenharia:* Esta métrica utiliza exclusivamente OHLC puro, pois o stream público OTC não fornece volume real; **não é VWAP volumétrica**.

### Pilar 4: Mitigação de Fair Value Gap (FVG)
- Identificação de desequilíbrio estrutural (imbalance) entre as velas $t-2$ e $t$.
- O recuo deve mitigar (penetrar e testar) a faixa de preço do FVG, confirmando o preenchimento de liquidez antes da continuação do movimento.

### Operação Contínua 24h (Sem Gate de Killzone)
- O antigo veto horário `fora_da_killzone_brt` foi removido por completo após validação em backtest, permitindo avaliação contínua em todas as velas M15 fechadas nos 12 índices OTC.

---

## 3. Janela de Execução e Validação Econômica

- **Janela Temporal:** A abertura do contrato deve ser disparada em até **60 segundos** após a abertura da nova vela M15. Transcorridos 60 segundos, a ordem é abortada por timeout (`SKIP_TIMEOUT`).
- **Validação de Valor Esperado (EV):**
  $$EV = P(	ext{Win}) 	imes (1 + 	ext{Payout}_{	ext{líquido}}) - 1$$
- Ordens em conta REAL exigem:
  1. $EV > 0$ calculado sobre o payout líquido confirmado pela proposta do broker.
  2. Margem de segurança de pelo menos $0,01$ acima do ponto de equilíbrio ($BE = rac{1}{1 + 	ext{Payout}}$).

---

## 4. Gestão de Risco e Dimensionamento

- **Stake Base e Cluster Risk Budget:** `base_stake_pct = 1,5%` (piso `1%`, teto absoluto `5%` por ordem) com orçamento total por ciclo `cycle_risk_budget_pct = 4%` dividido entre as `k` ordens simultâneas do ciclo (`max_orders_per_cycle = 4`).
- **Desativação da Recuperação 1/1 e do `stop_win` Curto:** `smooth_recovery = false`, `amort_steps = 0` e `compounding_enabled = false`, evitando amplificação de drawdown em regimes adversos e decapitação assimétrica de dias positivos.
- **Teto Inviolável:** Nenhuma ordem pode ultrapassar **5% da banca disponível**.
- **Banimento Estrito:** Martingale, D'Alembert e Kelly exponencial estão terminantemente proibidos da rota operacional.

---

## 5. Critérios de Promoção e Amostragem (LLN)

- **Piloto DEMO:** Novas implementações ou calibrações entram inicialmente em modo `piloto` na conta DEMO.
- **Amostra de 200 Contratos:** A promoção para conta REAL exige histórico de pelo menos **200 contratos únicos liquidados pelo broker** com a mesma versão de setup.
- **Limite Inferior Positivo:** O limite inferior unilateral de 95% do intervalo de confiança para o retorno líquido médio deve ser estritamente superior a zero ($IC_{95\%} > 0$).
