---
name: aether-binary-senior
description: >-
  Auditoria e execucao senior M5 em indices de acoes mundiais (OTC) e sinteticos com setup institucional (senior_institutional), VWAP, FVG e Killzones BRT.
---

# Playbook Senior Binario M5 (ETFs / OTC Mundiais)

Consulte `docs/binary-senior-playbook.md` e `docs/strategy-pullback.md`.

Pipeline Operacional: Setup Institucional M5 (`senior_institutional`) -> Validacao de Killzone BRT -> Confluencia EMA/ATR -> Session VWAP (+/- 1.5 sigma) -> Mitigacao FVG -> Quote Guard com EV Positivo -> Dimensionamento Institucional (1% banca, recovery max 5%).

## Checklist de Auditoria por Ciclo M5

1. **Horário BRT**: Está dentro de uma das 3 Killzones (London Open 03h-08h, NY Overlap 10h-13h ou London Fix 14h-16h BRT)? Se não, aborta com SEM SINAL.
2. **Estrutura de Médias**: EMA20 e EMA50 alinhadas com inclinação saudável e recuo penetrando até 0.5 ATR14 da EMA20.
3. **Session VWAP**: Fechamento favorável à média ponderada diária e estritamente contido no canal de $\pm 1.5$ desvios padrão.
4. **Fair Value Gap**: Desequilíbrio institucional mitigado na retração antes do rompimento.
5. **Cotação e EV**: Payout líquido cotado pela Deriv sustentando EV estritamente positivo e margem mínima sobre o break-even.
6. **Stake Institucional**: Base 1% da banca disponível. Recuperação linear 1/1 limitada a 5% por ordem.

## Proibições

- Operar fora das janelas de liquidez institucional.
- Aumentar stake acima de 5% da banca por trade.
- Utilizar Martingale, D'Alembert ou Kelly descontrolado.
- Reverter arbitrariamente o lado do sinal sem confluência da matriz de 5 pilares.
