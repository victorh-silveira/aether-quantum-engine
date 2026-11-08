---
name: aether-indicator-consensus
description: Auditoria do setup institucional senior de tendencia/recuo (senior_institutional) e gates M15.
---

# Auditoria do Setup Institucional M15

## 1. Contexto e Pré-requisitos
- Leia `AGENTS.md`, `docs/strategy-pullback.md` e `.agents/rules/aether-indicator-consensus.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Auditoria de Séries Temporais:**
   - Confirme que EMA20, EMA50, ATR14, referencia OHLC da sessao UTC e FVG usem exclusivamente velas M15 fechadas, sem lacunas intradiarias.
2. **Auditoria de Versão do Setup:**
   - Verifique se a versao exata (`senior_institutional`) coincide no sinal, no registro de cotacao e na proposta.
3. **Verificação de Killzones:**
   - Assegure que entradas ocorram estritamente em: London Open (03h-08h), New York Overlap (10h-13h) ou London Fix (14h-16h) no Horario de Brasilia.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/domain/test_indicator_consensus*.py -v
```

## 4. Critérios de Aceite
- Setup estritamente alinhado com a especificacao matematica institucional.
- Zero ordens abertas fora dos limites autorizados.
