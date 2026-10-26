---
name: aether-indicator-consensus
description: Audita o setup institucional de tendencia/recuo (senior_institutional), referencia OHLC da sessao, FVG, Killzones BRT e promocao por contratos DEMO.
---

# Setup Institucional de Tendencia e Recuo

1. Leia `AGENTS.md`, `indicator/AGENTS.md` e `docs/strategy-pullback.md`.
2. Confirme que EMA20/EMA50, ATR14, referencia OHLC da sessao UTC (+/- 1.5 sigma), mitigacao de Fair Value Gap (FVG) e Hard Gate de Killzones no Horario de Brasilia usam somente velas fechadas, atuais e sem lacunas intradiarias por ativo e periodo.
3. Confira versao exata do setup (`senior_institutional`) no sinal, registro, cotacao, compra e liquidacao. O proxy de velas e diagnostico; ele nao qualifica contratos. Nao reutilize o historico de modelos ML. O treino nao promove contratos nem altera settings.
4. DEMO: primeira amostra de ate 200 setups M15 com proposta disponivel, stake base de 1% e sem filtro de EV. Contar somente IDs unicos liquidados pelo broker. REAL: limite inferior unilateral de 95% positivo, posterior Beta(0,5;0,5), haircut, EV positivo e margem de 0,01 na proposta final.
5. Preserve 60 segundos de entrada, duracao M15 de 15 minutos, base de 1% da banca disponivel e recuperacao por payout cotado ate 5% por ordem. H1/D1 ficam no treino e pesquisa. Confira deduplicacao, compra sem ACK e suspensao apos 200 liquidacoes REAL com limite nao positivo.
6. Execute `make app-lint`, `make app-test`, seguranca, build e smoke sem iniciar compras REAL.
