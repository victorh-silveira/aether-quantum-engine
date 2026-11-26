# Estrategia Continua Laya System 1 (`R_10` M5)

## 1. Visao Geral

Nesta branch (`feat/laya-continuous-r10`), o motor substitui todas as heuristicas, vetos e modelos TCN legados por inferencia direta e continua via **Laya System 1** (`laya==0.5.5`, `convaiinnovations/laya`, `vegaml=False`):

- **Ativo:** `R_10` (Volatility 10 Index — operacao 24/7).
- **Timeframe e Contrato:** velas fechadas **M5 (`300s`)**, contratos Rise/Fall de **5 minutos** (`duration=5`, `duration_unit="m"`).
- **Politica Zero SKIP:** todo ciclo M5 gera obrigatoriamente uma decisao `CALL` ou `PUT`. Em caso de empate exato (`P(CALL) == P(PUT)`), aplica-se desempate deterministico (`ema_20 >= ema_50 -> CALL`, caso contrario `PUT`).

---

## 2. Vetor Completo de Indicadores Enviados ao Laya

Calculado em `src/domain/indicators/technical_indicators.py` sobre velas M5 fechadas:

- **Tendencia:** `ema_9`, `ema_20`, `ema_21`, `ema_50`, `ema_spread_atr`, `ema20_slope`, `ema50_slope`.
- **Momento:** `rsi_7`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `macd_hist_delta`, `stoch_k`, `stoch_d`, `cci_14`, `cmo_14`, `williams_r_14`, `roc_14`.
- **Direcionalidade:** `adx_14`, `plus_di_14`, `minus_di_14`, `di_diff_14`, `adx_20`, `di_diff_20`.
- **Volatilidade e Bandas:** `atr_14`, `atr_50`, `atr_ratio`, `bollinger_pct_b`, `bollinger_width`, `keltner_pct_b`.
- **Microestrutura e Regime:** `hurst_14`, `body_atr`, `upper_wick_atr`, `lower_wick_atr`, `candle_dir`, `streak`, `session_zscore`.

Alem dos indicadores, o adapter envia as ultimas 12 velas OHLC fechadas e as configuracoes sanitizadas de `config/settings.json` (sem chaves ou tokens).

O contexto e serializado de forma compacta, com indicadores antes das velas. O orcamento inclui pergunta, opcoes, estado e separadores; `max_len` e calculado com o tokenizer real e limitado pela capacidade do encoder (8.192 posicoes no checkpoint atual). O limite padrao de 512 tokens nao e usado para cortar o contexto. Respostas com `usage.truncated` ou tokens descartados sao rejeitadas como erro tecnico. Os scores de escolha e confianca nao comprovam calibracao de acerto financeiro.

A verificacao offline pode ser reproduzida com `python -m tools.laya_inference_probe` no runtime Conda `deriv-api`, via WSL. Ela exige pesos locais, compara cenarios sinteticos e nao conecta um executor de ordens.

---

## 3. Dimensionamento de Stake (`1%` a `5%`)

Implementado em `src/domain/risk/kelly_recovery.py`, adaptando formulas da main `460c4f41`:

- Kelly bruto `f=(b*p-(1-p))/b`; aplicar fracao 0,25 somente a parte positiva. `b` e o lucro liquido da proposta dividido pelo preco.
- Piso de 1% e minimo $1,00 mesmo quando Kelly nao recomenda aposta, por decisao do operador; registrar origem `FLOOR`, `KELLY` ou `RECOVERY`.
- A perda pendente aumenta pelo loss realizado e diminui pelo lucro liquido dos wins. Win parcial nao zera divida.
- Etapas `ceil(pending/(payout*cap))`; parcela `pending/etapas`. Stake final e o maior valor entre Kelly com piso e parcela/payout, limitada a 5% e arredondada sem ultrapassar o teto.
- Revalidar saldo, preco, payout e dimensionamento na proposta usada para compra WSS. REST sem proposta final verificavel nao executa esse fluxo.
- Persistir estado financeiro e IDs liquidados; nenhuma liquidacao duplicada reaplica a divida.

O modelo atual permanece ativo ate existir checkpoint local aprovado. O fluxo GPU e os criterios de promocao estao em [laya-training.md](laya-training.md). Nao ha promessa de rentabilidade ou eliminacao de losses.
