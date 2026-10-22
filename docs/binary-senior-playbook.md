# Playbook Trader Senior — Contratos Rise/Fall M5 (ETFs / OTC Mundiais)

## 1. Postura e Filosofia Operacional

O Aether Quantum Engine opera exclusivamente como um algoritmo de arbitragem estatística e confluência institucional de fluxo de ordens. Não há adivinhação, intuição ou apostas agressivas. A execução é regida pela variante **`senior_institutional`** em velas M5 fechadas (contrato de 300 segundos).

- **Ativo Promovido Vigente**: **`OTC_SX5E`** (Euro Stoxx 50 / FEZ), selecionado por diagnóstico de EV esperado positivo (+0.0338), assertividade de 61.54% e máximo de apenas 3 perdas consecutivas.
- **Universo de Monitoramento**: 12 índices sintéticos mundiais de ações OTC integrados a feeds reais do Yahoo Finance v8 (OTC_SPC, OTC_NDX, OTC_DJI, OTC_GDAXI, OTC_FTSE, OTC_N225, OTC_SX5E, OTC_SSMI, OTC_FCHI, OTC_AEX, OTC_HSI e OTC_AS51).
- **Tempo e Cadência**: Ciclo rigoroso de 300 segundos (M5). Abertura de ordens autorizada somente até 30 segundos após a abertura da nova vela.

---

## 2. Checklist Mental do Trader Sênior (Matriz de 5 Pilares)

Antes de autorizar o envio de qualquer proposta à corretora, o motor valida 5 condições independentes:

```
[1] HORÁRIO BRT (Killzone Ativa?)
    └── London Open (03h–08h) OU NY Overlap (10h–13h) OU London Fix (14h–16h)
        └── SE NÃO ➔ VETO (SEM SINAL: Fora da Killzone)

[2] TENDÊNCIA ESTRUTURAL (EMA20 > EMA50?)
    └── Alinhamento de médias e inclinação positiva
        └── SE NÃO ➔ VETO (SEM SINAL: Mercado Lateral ou Indeciso)

[3] RETRAÇÃO DE VOLATILIDADE (Pullback Saudável?)
    └── Recuo penetra até 0.5 ATR14 da EMA20 sem violar a EMA50
        └── SE NÃO ➔ VETO (SEM SINAL: Retração Excessiva ou Insuficiente)

[4] SESSION VWAP & BANDAS (Preço Institucional Favorável?)
    └── Fechamento favorável à VWAP diária e contido em +/- 1.5 sigma
        └── SE NÃO ➔ VETO (SEM SINAL: Exaustão Institucional Contra a VWAP)

[5] DESEQUILÍBRIO INSTITUCIONAL (Fair Value Gap Mitigado?)
    └── Imbalance entre t-2 e t testado e fechado pela retração
        └── SE NÃO ➔ VETO (SEM SINAL: Falta de Mitigação de FVG)
```

---

## 3. Gestão de Risco e Dimensionamento de Capital

1. **Stake Base**: Rigorosamente **1% da banca disponível** (em DEMO e REAL).
2. **Sem Martingale**: É terminantemente proibido dobrar lotes após perdas.
3. **Amortização Linear 1/1**: Em caso de perda, a recuperação calcula a stake necessária baseando-se no payout líquido cotado:
   $$Stake_{rec} = \min\left(\frac{Perda}{Payout}, 5\% \times Banca\right)$$
   com teto máximo inviolável de **5% por ordem**.
4. **Veto Econômico na Cotação Final**:
   $$EV = p \times (1 + Payout_{líquido}) - 1 > 0$$
   com margem mínima de 0,01 acima do break-even. Se a Deriv oferecer payout degradado que negative o EV, a ordem é abortada no último milissegundo.

---

## 4. Auditoria de Sessão e Comandos Operacionais

- Treinamento e Diagnóstico de Ativos:
  ```bash
  make app-train
  ```
- Execução Operacional:
  ```bash
  make app-run
  ```
- Auditoria de Qualidade e Pré-commit:
  ```bash
  make app-pre-commit-run
  ```
- Monitoramento Live:
  Dashboards Grafana em `http://localhost:3000` (Painéis Aether e Sintéticos).
