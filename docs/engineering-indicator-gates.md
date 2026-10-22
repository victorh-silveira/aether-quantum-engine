# Catalogo de gates e filtros de execucao (SSOT)

## 1. Politica Vigente: Matriz Institucional Senior (senior_institutional)

Quando `pullback_strategy.enabled=true` e `pullback_variant="senior_institutional"`, o fluxo de autorizacao de trades obedece a matriz de confluencia de 5 pilares institucionais:

| Gate Institucional | Regra de Validacao | Acao em Falha |
|---|---|---|
| `gate_killzone_brt` | Timestamp M5 deve estar dentro de uma das janelas de liquidez global convertidas para o Horário de Brasília (UTC-3): London Open (03h–08h), New York Overlap (10h–13h) ou London Fix (14h–16h). | Emite `SEM SINAL` com motivo `Fora da killzone institucional de liquidez`. |
| `gate_session_vwap_trend` | Fechamento da confirmacao deve estar a favor da VWAP da sessao diaria (00:00 UTC): $Close \ge VWAP$ para CALL e $Close \le VWAP$ para PUT. | Emite `SEM SINAL` com motivo `Fechamento contra a Session VWAP institucional`. |
| `gate_session_vwap_band` | Fechamento contido no canal de dispersao de $\pm 1.5\sigma$: $Close \le VWAP + 1.5\sigma$ para CALL e $Close \ge VWAP - 1.5\sigma$ para PUT. | Emite `SEM SINAL` com motivo `Exaustao alem da banda de dispersao VWAP 1.5 sigma`. |
| `gate_fvg_mitigation` | Identifica Fair Value Gap entre velas $t-2$ e $t$ e exige que a vela de recuo tenha penetrado a regiao de desequilibrio antes da expansao. | Emite `SEM SINAL` com motivo `Ausencia de mitigacao de Fair Value Gap (FVG)`. |
| `gate_quote_edge` | A cotacao final retornada pelo broker Deriv deve ter preco e payout liquidos validos, com expectativa matematica positiva: $EV = p \times (1 + payout) - 1 > 0$ e margem minima de 0,01 sobre o break-even. | Bloqueia o envio do contrato por falta de borda economica. |

---

## 2. Politica de Quatro Vetos de Mercado Extremos (Legado / Respaldo)

Para ativos onde nao ha sinal institucional formado, `orchestrator.execution.four_market_vetoes=true` mantem a sentinela contra exaustao extrema de preco:

| Veto | Confluencia na ultima vela fechada |
|------|------------------------------------|
| `call_top_rejection` | RSI >= 0.68 e BB %B >= 0.88 com pavio superior >= 30%, ou vela de exaustao vendedora no extremo |
| `call_down_continuation` | Corpo vendedor >= 70%, DI diff <= -0.12, tendencia e vela anterior PUT |
| `put_bottom_rejection` | RSI <= 0.32 e BB %B <= 0.12 com pavio inferior >= 30%, ou vela de exaustao compradora no extremo |
| `put_up_continuation` | Corpo comprador >= 70%, DI diff >= 0.12, tendencia e vela anterior CALL |

---

## 3. Hot Path Operacional Sênior

1. **SKIP Tecnico**: Falha de conexao WSS, dados insuficientes de historico (< 128 velas fechadas), descontinuidade ou pausa operacional.
2. **Avaliacao Institucional M5**:
   - Validacao de Killzone BRT (03h-08h, 10h-13h, 14h-16h).
   - Validacao de Tendencia e Inclinacao (EMA20 > EMA50).
   - Validacao de Retracao (recuo ate 0,5 ATR14).
   - Validacao de Volume e Dispersao (Session VWAP $\pm 1.5\sigma$).
   - Validacao de Desequilibrio (FVG mitigado).
3. **Bloqueio Economico Fail-Closed**: Preco, payout cotado pela Deriv e EV liquido verificado. Se EV <= 0 ou payout ausente, veto imediato.
4. **Dimensionamento de Capital**: Stake base de 1% da banca disponivel. Se houver perda pendente, amortizacao linear 1/1 com teto rigoroso de 5% por ordem.
5. **Deduplicacao e Quarentena**: `SET NX` no Redis por contrato/vela; quarentena de 900 s em timeout de compra sem ACK.
