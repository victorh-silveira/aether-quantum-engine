---
name: aether-binary-senior
description: Auditoria dos gatilhos persistent_trend_v3 e da execucao Rise/Fall M15 nos 12 ativos OTC, preservando gates e risco vigentes.
---

# Execução Sênior do Setup Institucional M15

## 1. Contexto e Pré-requisitos
- Leia `AGENTS.md`, `docs/strategy-pullback.md` e `docs/binary-senior-playbook.md`.
- Ativo ancora: `OTC_FTSE` com rotacao nos 12 ativos OTC (contratos Rise/Fall M15 com duracao de 900s).
- Variante vigente: `persistent_trend_v3`; `senior_institutional`, `dual_sniper` e `multi_regime_hf` permanecem disponiveis para comparacao e rollback.

## 2. Procedimento Operacional Passo a Passo
1. **Auditoria dos Ramos Causais na Vela Fechada:**
   - Verifique os ramos B1/B2 de alinhamento H1 EMA20/50 e Wilder DI(24)/DI(14), B3 de recuo DI(14), B4 de continuacao DI(28) e B5 DI(20) com ADX(20), conforme `docs/strategy-pullback.md`.
   - Preserve o veto de conflito e as horas toxicas UTC 03h/06h/19h. H1 deve incluir apenas velas fechadas e ser atualizado antes da avaliacao M15 na virada da hora.
   - Aplique `EconomicCalendarGate` e distinga cobertura historica comprovada de um cache semanal.
   - Para pesquisa de frequencia, leia `docs/strategy-frequency-research.md`; proxies OHLC nao qualificam contratos REAL.
2. **Verificação da Janela de Entrada:**
   - Confirme se o timestamp atual esta dentro dos primeiros 60 segundos da abertura da vela M15 atual. Caso exceda 60s, acione SKIP_TIMEOUT.
3. **Validação Financeira:**
   - Requisite proposta a Deriv, verifique se o payout liquido e valido e se o EV calculado e positivo com margem >= 0,01.
   - Preserve base de 1,5%, orcamento de 4% por ciclo, quatro ordens no maximo, piso de 1% e teto absoluto de 5% por ordem, sem recuperacao de perdas.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest tests/unit/application/test_pullback_execution_flow.py tests/unit/domain/risk/test_pullback_stake.py -v
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest tests/indicator/test_strategy_persistent_trend.py tests/indicator/test_frequency_candidate.py tests/indicator/test_frequency_h1_refresh.py -v
```

## 4. Critérios de Aceite
- Entrada estrita sob candle anterior fechado.
- Rejeicao automatica de qualquer proposta com EV nao positivo.
