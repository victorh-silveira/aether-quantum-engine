# Estratégia Quantitativa: Pullback Institucional M15

## 1. Visão Geral da Estratégia

O Aether Quantum Engine adota a estratégia quantitativa de recuo institucional (`senior_institutional`) operando contratos Rise/Fall de 15 minutos (900 segundos) na corretora Deriv.

O ativo âncora soberano é o **`OTC_SSMI`** (Swiss Market Index 20 / Deriv OTC). O universo de candidatos avaliados em pesquisa pelo subsistema `indicator/` abrange 12 ativos mundiais de ações OTC (OTC_SPC, OTC_NDX, OTC_DJI, OTC_GDAXI, OTC_FTSE, OTC_N225, OTC_SX5E, OTC_SSMI, OTC_FCHI, OTC_AEX, OTC_HSI e OTC_AS51).

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

### Pilar 5: Hard Gate de Killzones no Horário de Brasília (BRT / UTC-3)
A entrada só é autorizada se o fechamento da vela de confirmação estiver contido estritamente em uma das 3 janelas de maior liquidez institucional:
1. **London Open:** 03h00 às 08h00 BRT
2. **New York Overlap:** 10h00 às 13h00 BRT
3. **London Fix:** 14h00 às 16h00 BRT
Qualquer vela fora desses intervalos recebe veto sumário (`SEM SINAL: Fora da Killzone`).

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
