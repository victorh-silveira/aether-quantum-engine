-- Auditoria de contratos: tabela pequena, indexada por id, sem retencao automatica.
-- Campos de spot ficam NULL quando a corretora nao os forneceu.
CREATE TABLE IF NOT EXISTS contract_executions (
  contract_id BIGINT PRIMARY KEY,
  symbol TEXT NOT NULL,
  account_mode TEXT NOT NULL,
  direction TEXT NOT NULL,
  transaction_buy_id TEXT,
  request_epoch_ms BIGINT,
  ack_epoch_ms BIGINT,
  date_start BIGINT,
  date_expiry BIGINT,
  entry_tick DOUBLE PRECISION,
  entry_tick_time BIGINT,
  exit_tick DOUBLE PRECISION,
  exit_tick_time BIGINT,
  buy_price DOUBLE PRECISION,
  payout DOUBLE PRECISION,
  signal_prob DOUBLE PRECISION,
  profit DOUBLE PRECISION,
  status TEXT,
  settlement_source TEXT NOT NULL DEFAULT 'pending',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS signal_prob DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS contract_executions_symbol_start
  ON contract_executions (symbol, date_start DESC);

-- Apenas contratos com spots e lucro confirmados pelo broker entram na comparacao.
-- A vela de entrada e a ultima M5 fechada antes do envio da compra.
CREATE OR REPLACE VIEW contract_label_audit AS
SELECT c.contract_id, c.symbol, c.account_mode, c.direction,
       c.request_epoch_ms, c.ack_epoch_ms, c.date_start, c.date_expiry,
       c.entry_tick, c.entry_tick_time, c.exit_tick, c.exit_tick_time,
       c.buy_price, c.payout, c.profit, c.status,
       b0.epoch AS m5_entry_epoch, b0.close AS m5_entry_close,
       b1.epoch AS m5_exit_epoch, b1.close AS m5_exit_close,
       CASE c.direction WHEN 'CALL' THEN b1.close > b0.close
                        WHEN 'PUT' THEN b1.close < b0.close END AS m5_win,
       CASE c.direction WHEN 'CALL' THEN c.exit_tick > c.entry_tick
                        WHEN 'PUT' THEN c.exit_tick < c.entry_tick END AS spot_win,
       c.profit > 0 AS broker_win,
       c.signal_prob,
       CASE WHEN c.signal_prob BETWEEN 0 AND 1 THEN
         power(c.signal_prob - (c.exit_tick > c.entry_tick)::int, 2)
       END AS broker_brier
FROM contract_executions c
JOIN LATERAL (
  SELECT epoch, close FROM ohlc_bars
  WHERE symbol=c.symbol AND granularity=300 AND epoch % 300=0
    AND epoch + 300 <= c.request_epoch_ms / 1000
  ORDER BY epoch DESC LIMIT 1
) b0 ON true
JOIN ohlc_bars b1 ON b1.symbol=c.symbol AND b1.granularity=300
  AND b1.epoch=b0.epoch + 300 AND b1.epoch % 300=0
WHERE c.settlement_source='broker' AND c.request_epoch_ms IS NOT NULL
  AND c.entry_tick IS NOT NULL AND c.exit_tick IS NOT NULL
  AND c.entry_tick_time IS NOT NULL AND c.exit_tick_time IS NOT NULL
  AND c.profit IS NOT NULL AND c.buy_price > 0;
