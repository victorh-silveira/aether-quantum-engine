---
name: aether-cycle-debug
description: Depuracao de ciclo do orquestrador, warmup de buffers, assinaturas de ciclo e watchdog.
---

# Depuração de Ciclos e Orquestração

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-orchestrator.md` e a regra `.agents/rules/aether-orchestrator.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Análise de Warmup:**
   - Inspecione se `get_closed_candles(symbol, 300)` retorna pelo menos 51 velas M5 fechadas, ordenadas e unicas, sem fallback diario.
2. **Inspeção de Assinaturas de Ciclo:**
   - Monitore a geracao de assinaturas (epoch + symbol) para verificar se algum ciclo foi pulado ou executado em duplicidade.
3. **Integridade do Contexto:**
   - Conferir `LAYA_CONTEXT` com epoch, assinatura e zero tokens descartados. Preservar a serie capturada antes da inferencia ate a persistencia. Usar `python -m tools.laya_inference_probe` para verificar o checkpoint local sem enviar ordens.
4. **Auditoria de Watchdog:**
   - Inspecione se o watchdog esta registrando batimentos regulares (heartbeats) e se nao ha deadlocks assincronos.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest tests/unit/application/test_laya_m5_pipeline.py tests/unit/infrastructure/test_stream_closed_candles.py tests/unit/infrastructure/test_laya_context.py -v
```

## 4. Critérios de Aceite
- Ciclos M5 sem repetir a mesma vela apos liquidacao ou reconexao.
- CALL/PUT continuo sem SKIP; falhas tecnicas de dados ou contexto identificadas explicitamente.
