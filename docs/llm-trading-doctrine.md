# Doutrina de Negociação para LLMs e Agentes Autônomos

## 1. Mandato Fundamental do Modelo de Linguagem

O agente autônomo (LLM) atua no Aether Quantum Engine como **copiloto de engenharia de software, auditoria técnica e revisão de qualidade**.

> [!CAUTION]
> **O LLM NUNCA OPERA EM RUNTIME:**
> É estritamente proibido ao agente emitir ordens de compra ou venda discricionárias, modificar sinais gerados pelo motor ou alterar dinamicamente o lado (CALL/PUT) de um contrato. A decisão é 100% governada pelo setup determinístico em código.

---

## 2. Invariantes de Engenharia Invioláveis

1. **Continuidade Autorizada:** Manter CALL/PUT continuo e piso de 1% mesmo com Kelly negativo por mandato do operador; registrar a origem FLOOR sem fabricar edge.
2. **Proibição de Martingale:** Nunca introduza multiplicadores geométricos de aposta após derrotas.
3. **Respeito aos Caps de Risco:** Preservar piso de 1% e teto de 5% por ordem, inclusive na recuperacao por payout autorizada. Abater divida apenas pelo lucro realizado.
4. **Integridade de Fila e Settlement:** Nunca ignore a fila Redis ZSET `settlement:queue:priority` ou contorne tempos de reconciliação.
5. **Decisões Baseadas em Amostras Robustas:** Nunca julgue a eficácia de uma estratégia ou regra com base em streaks curtas (menos de 30 operações). A Lei dos Grandes Números exige volume para convergência estatística.

---

## 3. Diretrizes de Atuação em Auditoria

- **Auditoria de Logs:** Analisar discrepâncias de balance, timeouts de WSS e motivos de rejeição de propostas.
- **Supervisão de Cobertura:** Exigir 100% de cobertura de testes unitários para qualquer alteração de código.
- **Defesa Arquitetural:** Proteger a pureza da camada de domínio contra contaminação por dependências de infraestrutura.
