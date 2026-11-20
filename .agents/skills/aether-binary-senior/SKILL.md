---
name: aether-binary-senior
description: Auditoria e execucao senior M15 nos 12 ativos OTC com setup senior_institutional (4 pilares em operacao 24h).
---

# Execução Sênior do Setup Institucional M15

## 1. Contexto e Pré-requisitos
- Leia `AGENTS.md`, `docs/strategy-pullback.md` e `docs/binary-senior-playbook.md`.
- Ativo ancora: `OTC_SSMI` com rotacao nos 12 ativos OTC (contratos Rise/Fall M15 com duracao de 900s).

## 2. Procedimento Operacional Passo a Passo
1. **Auditoria dos 4 Pilares na Vela Fechada (Operacao 24h):**
   - Verifique o alinhamento da tendencia EMA20/EMA50 na vela anterior fechada.
   - Calcule a distancia de recuo da vela anterior em relacao a EMA20 e ATR14.
   - Avalie o preco em relacao a referencia da sessao UTC: deve estar contido entre as bandas de +/- 1,5 desvio padrao.
   - Identifique a mitigacao de Fair Value Gap (FVG) entre $t-2$ e $t$.
2. **Verificação da Janela de Entrada:**
   - Confirme se o timestamp atual esta dentro dos primeiros 60 segundos da abertura da vela M15 atual. Caso exceda 60s, acione SKIP_TIMEOUT.
3. **Validação Financeira:**
   - Requisite proposta a Deriv, verifique se o payout liquido e valido e se o EV calculado e positivo com margem >= 0,01.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/application/test_pullback_execution_flow.py -v
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/domain/test_pullback*.py -v
```

## 4. Critérios de Aceite
- Entrada estrita sob candle anterior fechado.
- Rejeicao automatica de qualquer proposta com EV nao positivo.
