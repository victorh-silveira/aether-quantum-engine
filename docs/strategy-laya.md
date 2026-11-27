# Estrategia Continua Laya System 1 (`R_10` M5)

## 1. Visao Geral

Nesta branch (`feat/laya-continuous-r10`), o motor substitui todas as heuristicas, vetos e modelos TCN legados por inferencia direta e continua via **Laya System 1** (`laya==0.5.5`, `convaiinnovations/laya`, `vegaml=False`):

- **Ativo:** `R_10` (Volatility 10 Index — operacao 24/7).
- **Timeframe e Contrato:** velas fechadas **M5 (`300s`)**, contratos Rise/Fall de **5 minutos** (`duration=5`, `duration_unit="m"`).
- **Politica de SKIP Configuravel:** Todo ciclo M5 gera uma decisao `CALL` ou `PUT` pelo Laya (em caso de empate exato `P(CALL) == P(PUT)`, aplica-se desempate deterministico `ema_20 >= ema_50 -> CALL`, senao `PUT`). Em seguida, `laya.skip_policy` em `config/settings.json` (`enabled`, `min_probability`, `min_body_atr`, `block_counter_momentum`, `block_strong_counter_trend`, `max_counter_streak`) decide antes da cotacao na Deriv se o ciclo e executado ou registrado como `SKIP | ...`.

---

## 2. Vetor Completo de Indicadores e Narrativa Enviada ao Laya

Calculado em `src/domain/indicators/technical_indicators.py` sobre velas M5 fechadas (janela causal de ate 120 velas):

- **Tendencia:** `ema_9`, `ema_20`, `ema_21`, `ema_50`, `ema_spread_atr`, `ema20_slope`, `ema50_slope`.
- **Momento:** `rsi_7` e `rsi_14` (suavizacao de Wilder EWM), `macd`, `macd_signal`, `macd_hist`, `macd_hist_delta`, `stoch_k`, `stoch_d`, `cci_14`, `cmo_14`, `williams_r_14`, `roc_14`.
- **Direcionalidade:** `adx_14`, `plus_di_14`, `minus_di_14`, `di_diff_14`, `adx_20`, `di_diff_20`.
- **Volatilidade e Bandas:** `atr_14`, `atr_50`, `atr_ratio`, `bollinger_pct_b`, `bollinger_width` (exibida em `%` no log como `BBw`), `keltner_pct_b`.
- **Microestrutura e Regime:** `hurst_14` (calculado com janela R/S de ate 50 barras), `body_atr`, `upper_wick_atr`, `lower_wick_atr`, `candle_dir`, `streak`, `session_zscore`.

Em `src/infrastructure/inference/laya_narrative.py`, os indicadores sao traduzidos para uma narrativa tecnica concisa em ingles (~370–440 tokens totais no contexto) contendo:
- **Estrutura de tendencia em 3 estados:** `bullish`, `bearish` ou `mixed/transition` com posicao explicita do fechamento em relacao as medias (`price above/below/between EMA9/EMA20`).
- **Regime, momento, volatilidade e pavios:** `ADX14`, `dDI`, `Hurst`, `MACD hist/dHist`, `RSI7/14`, `Stoch`, `CCI`, `BB%B`, `ATR` + `atr_ratio` (`expanding`/`compressed`/`normal`), ultimas 6 velas fechadas, `streak`, `body_atr`, `upper/lower_wick_atr` e `session_zscore`.

A inferencia executa duas passadas simetricas (`CALL/PUT` e `PUT/CALL`) e usa a media exata para neutralizar vies posicional. O `max_len` e dimensionado pelo tokenizer real dentro da capacidade do encoder; respostas com `usage.truncated` ou tokens descartados sao rejeitadas como erro tecnico.

---

## 3. Dimensionamento de Stake (`kelly_recovery`) e Stop-Win

Implementado em `src/domain/risk/kelly_recovery.py`, adaptado da main `460c4f41`:

- Kelly bruto `f=(b*p-(1-p))/b`; aplicar `risk_management.kelly.fraction` (atual `0.05`) somente a parte positiva, limitado pelo teto base `risk_management.kelly.max_base_pct` (atual `0.0125` = `1.25%`).
- Piso continuo de `1.00%` e minimo `$1.00` mesmo quando Kelly nao recomenda aposta (`FLOOR`).
- A perda pendente (`pending_loss`) aumenta pelo loss realizado e diminui pelo lucro liquido dos wins. Win parcial nao zera divida.
- Etapas `ceil(pending/(payout*cap))`; parcela `pending/etapas`. Stake final e o maior valor entre Kelly com piso e parcela/payout, limitada ao teto absoluto de `5.00%` (`RECOVERY`).
- Revalidacao final na proposta WSS (`resize_laya_proposal`): sincroniza o saldo real `{"balance": 1}` e o `balance_after` da liquidacao sem travas de diferenca, enquanto o **Stop-Win** (`risk_management.params.stop_win_pct = 0.05`) e avaliado exclusivamente pelo lucro realizado da sessao (`realized_session_pnl`).

O modelo base permanece ativo ate existir checkpoint local aprovado conforme [laya-training.md](laya-training.md). Nao ha promessa de rentabilidade ou eliminacao de losses.
