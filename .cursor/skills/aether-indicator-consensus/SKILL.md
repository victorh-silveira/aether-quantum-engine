---
name: aether-indicator-consensus
description: Audita o setup fixo de tendencia/recuo e sua promocao por contratos DEMO.
---

# Setup de tendencia e recuo

1. Leia `AGENTS.md`, `indicator/AGENTS.md` e `docs/strategy-pullback.md`.
2. Confirme que EMA20/EMA50, ATR14, recuo e confirmacao usam somente velas fechadas, contiguas e atuais por indice e periodo.
3. Confira versao fixa do setup no sinal, registro, cotacao, compra e liquidacao. O proxy de velas e diagnostico; ele nao qualifica contratos. Nao reutilize o historico de modelos ML.
4. DEMO: primeira amostra de ate 200 setups M5 com proposta disponivel, stake base de 1% e sem filtro de EV. Contar somente IDs unicos liquidados pelo broker. REAL: limite inferior unilateral de 95% positivo, posterior Beta(0,5;0,5), haircut, EV positivo e margem de 0,01 na proposta final.
5. Preserve 30 segundos de entrada, duracao M5 de 5 minutos, base de 1% da banca disponivel e recuperacao por payout cotado ate 5% por ordem. M15/H1 ficam no treino e pesquisa. Confira deduplicacao, compra sem ACK e suspensao apos 200 liquidacoes REAL com limite nao positivo.
6. Execute `make app-lint`, `make app-test`, seguranca, build e smoke sem iniciar compras REAL.
