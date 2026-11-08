---
name: aether-deriv-connect
description: Diagnostico de conexao WebSocket Deriv, handshake WSS, autenticacao PAT/OTP e resiliencia.
---

# Conectividade WebSocket Deriv API

## 1. Contexto e Pré-requisitos
- Leia `docs/deriv-api-aether.md` e `.agents/rules/aether-deriv-api.mdc`.
- Tenha as variaveis de ambiente de teste (`DERIV_API_TOKEN` ou credenciais mockadas) configuradas.

## 2. Procedimento Operacional Passo a Passo
1. **Validação de Handshake:**
   - Verifique a resolucao DNS e conexao SSL segura com `wss://api.derivws.com/websockets/v3?app_id=...`.
2. **Autenticação:**
   - Envie requisicao `authorize` com token PAT e valide a resposta recebida (validacao de moeda, saldo e restricoes de conta).
3. **Monitoramento de Latência:**
   - Meça o Round-Trip Time (RTT) de mensagens `ping` a cada 30 segundos.
4. **Simulação de Queda e Reconexão:**
   - Verifique o comportamento sob desconexao abrupta: o socket deve tentar reconectar com backoff exponencial mantendo o estado local integro.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe scripts/operations/deriv_pat_connect.py
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/infrastructure/test_deriv_ws*.py -v
```

## 4. Critérios de Aceite
- Autenticacao segura sem vazamento de tokens nos logs.
- Reconexao automatica transparente em menos de 10 segundos.
