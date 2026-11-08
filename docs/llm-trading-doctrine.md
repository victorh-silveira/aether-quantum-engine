# Doutrina de Negociação para LLMs e Agentes Autônomos

## 1. Mandato Fundamental do Modelo de Linguagem

O agente autônomo (LLM) atua no Aether Quantum Engine como **copiloto de engenharia de software, auditoria técnica e revisão de qualidade**.

> [!CAUTION]
> **O LLM NUNCA OPERA EM RUNTIME:**
> É estritamente proibido ao agente emitir ordens de compra ou venda discricionárias, modificar sinais gerados pelo motor ou alterar dinamicamente o lado (CALL/PUT) de um contrato. A decisão é 100% governada pelo setup determinístico em código.

---

## 2. Invariantes de Engenharia Invioláveis

1. **Proibição de Forçar Negociações:** Nunca sugira ou ative `force_trade_every_cycle=true` para maquiar ausência de sinais.
2. **Proibição de Martingale:** Nunca introduza multiplicadores geométricos de aposta após derrotas.
3. **Respeito aos Caps de Risco:** Nunca aumente o limite de stake base de 1% ou o teto de 5% de recuperação.
4. **Integridade de Fila e Settlement:** Nunca ignore a fila Redis ZSET `settlement:queue:priority` ou contorne tempos de reconciliação.
5. **Decisões Baseadas em Amostras Robustas:** Nunca julgue a eficácia de uma estratégia ou regra com base em streaks curtas (menos de 30 operações). A Lei dos Grandes Números exige volume para convergência estatística.

---

## 3. Diretrizes de Atuação em Auditoria

- **Auditoria de Logs:** Analisar discrepâncias de balance, timeouts de WSS e motivos de rejeição de propostas.
- **Supervisão de Cobertura:** Exigir 100% de cobertura de testes unitários para qualquer alteração de código.
- **Defesa Arquitetural:** Proteger a pureza da camada de domínio contra contaminação por dependências de infraestrutura.
