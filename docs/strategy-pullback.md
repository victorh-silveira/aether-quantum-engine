# Estratégia Quantitativa: Pullback Institucional M15

## 1. Visão Geral da Estratégia

O Aether Quantum Engine adota como variante ativa o **Portfolio Multi-Regime de Alta Frequência (`multi_regime_hf`, versão `pullback-1-4c9986919dfb`)** operando contratos Rise/Fall de 15 minutos (900 segundos) na corretora Deriv, preservando `dual_sniper` e `senior_institutional` para retrocompatibilidade e rollback.

O universo operacional abrange os **12 ativos mundiais de ações OTC** (Europa: `OTC_GDAXI`, `OTC_FCHI`, `OTC_FTSE`, `OTC_SX5E`, `OTC_AEX`, `OTC_SSMI`; EUA: `OTC_SPC`, `OTC_NDX`, `OTC_DJI`; Ásia-Pacífico: `OTC_N225`, `OTC_AS51`, `OTC_HSI`), com análise 100% causal em velas M15 fechadas e execução multi-ordem controlada (`multi_symbol_orders=true`, `max_orders_per_cycle=4`).

### Ramos Ortogonais do `multi_regime_hf` (Combinados com `veto_conflito`)
1. **Ramo 1A — Climax Global em Compressão (`setup_climax_global`):** fade de expansão de True Range (`TR / ATR14_Wilder > 1.5`) quando a compressão média do bloco regional é `< 0.95` e o consenso global na mesma direção é `>= 65%` (exceto hora 20 UTC).
2. **Ramo 1B — Exaustão de Fechamento de Hora (`setup_q2_cci14`):** na vela `:30` (`min_of_hour == 30`, exceto hora 20 UTC), reversão por `CCI(14) <= -120` (CALL) ou `>= +120` (PUT) com `ATR14/ATR50 < 1.0` e consenso regional `>= 66%`.
3. **Ramo 2 — Momentum Wilder Rápido + Pullback (`setup_di5_pullback`):** fora da virada `:45`, entrada a favor da pressão direcional `DI+(5) - DI-(5)` (`> +50` para CALL em vela de recuo negativa; `< -50` para PUT em vela de recuo positiva).
4. **Ramo 3 — Compressão Regional Profunda (`setup_comp_fade`):** fade de barra forte (`|body|/ATR14 >= 0.80`, `range/ATR14 >= 0.25`) sob compressão profunda (`ATR14/ATR50 < 0.85`) e consenso regional `>= 75%` (mínimo 3 pares, exceto hora 20 UTC).
- **Resolução de Conflitos:** se dois ramos dispararem lados opostos no mesmo `(symbol, epoch)`, o sinal é vetado (`conflito_multi_regime`).

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

- **Stake Base:** Exatamente **1% do saldo disponível da banca**.
- **Amortização Linear 1/1:** Em caso de perda, a recuperação busca recompor o prejuízo acumulado de forma linear:
  $$	ext{Stake}_{	ext{rec}} = \min\left(\max\left(rac{	ext{Prejuízo}}{	ext{Payout}}, 1\% 	imes 	ext{Banca}ight), 5\% 	imes 	ext{Banca}ight)$$
- **Teto Inviolável:** Nenhuma ordem pode ultrapassar **5% da banca disponível**.
- **Banimento Estrito:** Martingale, D'Alembert e Kelly exponencial estão terminantemente proibidos da rota operacional.

---

## 5. Critérios de Promoção e Amostragem (LLN)

- **Piloto DEMO:** Novas implementações ou calibrações entram inicialmente em modo `piloto` na conta DEMO.
- **Amostra de 200 Contratos:** A promoção para conta REAL exige histórico de pelo menos **200 contratos únicos liquidados pelo broker** com a mesma versão de setup.
- **Limite Inferior Positivo:** O limite inferior unilateral de 95% do intervalo de confiança para o retorno líquido médio deve ser estritamente superior a zero ($IC_{95\%} > 0$).
