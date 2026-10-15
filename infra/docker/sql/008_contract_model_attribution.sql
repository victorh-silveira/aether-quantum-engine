ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS model_version TEXT;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS calibrated_call_prob DOUBLE PRECISION;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS indicator_model_version TEXT;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS indicator_call_prob DOUBLE PRECISION;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS indicator_candle_epoch BIGINT;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS quote_price DOUBLE PRECISION;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS quote_payout DOUBLE PRECISION;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS quote_ev DOUBLE PRECISION;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS indicator_period_seconds INTEGER;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS quote_cohort TEXT;

CREATE TABLE IF NOT EXISTS indicator_quote_observations (
  observation_id UUID PRIMARY KEY,
  observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  account_mode TEXT NOT NULL,
  symbol TEXT NOT NULL,
  period_seconds INTEGER NOT NULL,
  candle_epoch BIGINT NOT NULL,
  model_version TEXT NOT NULL,
  side TEXT NOT NULL,
  proposal_id TEXT NOT NULL,
  side_probability DOUBLE PRECISION,
  quote_price DOUBLE PRECISION,
  quote_payout DOUBLE PRECISION,
  quote_ev DOUBLE PRECISION,
  cohort TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS indicator_quotes_pair_time
  ON indicator_quote_observations(symbol, period_seconds, observed_at DESC);

CREATE OR REPLACE VIEW contract_model_outcomes AS
SELECT model_version, symbol, account_mode, coalesce(contract_type, direction) AS contract_type,
       count(*) AS broker_settled_count,
       count(*) FILTER (WHERE profit > 0) AS wins,
       avg(profit / buy_price) AS mean_unit_return,
       avg(power(calibrated_call_prob - (exit_tick > entry_tick)::int, 2))
         FILTER (WHERE entry_tick IS NOT NULL AND exit_tick IS NOT NULL) AS calibrated_brier,
       count(*) FILTER (WHERE entry_tick IS NOT NULL AND exit_tick IS NOT NULL) AS brier_samples
FROM contract_executions
WHERE model_version IS NOT NULL
  AND calibrated_call_prob BETWEEN 0 AND 1
  AND settlement_source IN ('broker', 'profit_table')
  AND coalesce(contract_type, direction) IN ('CALL', 'PUT')
  AND buy_price > 0 AND profit IS NOT NULL
GROUP BY model_version, symbol, account_mode, coalesce(contract_type, direction);

CREATE OR REPLACE VIEW contract_consensus_outcomes AS
SELECT model_version, indicator_model_version, symbol, account_mode, direction,
       count(*) AS broker_settled_count,
       count(*) FILTER (WHERE profit > 0) AS wins,
       avg(profit / buy_price) AS mean_unit_return,
       avg(power(indicator_call_prob - (exit_tick > entry_tick)::int, 2))
         FILTER (WHERE entry_tick IS NOT NULL AND exit_tick IS NOT NULL) AS indicator_brier,
       count(*) FILTER (WHERE entry_tick IS NOT NULL AND exit_tick IS NOT NULL) AS brier_samples,
       avg(quote_ev) AS mean_quoted_ev
FROM contract_executions
WHERE model_version IS NOT NULL
  AND indicator_model_version IS NOT NULL
  AND indicator_call_prob BETWEEN 0 AND 1
  AND settlement_source IN ('broker', 'profit_table')
  AND coalesce(contract_type, direction) IN ('CALL', 'PUT')
  AND buy_price > 0 AND profit IS NOT NULL
GROUP BY model_version, indicator_model_version, symbol, account_mode, direction;

DROP VIEW IF EXISTS indicator_broker_ev_status;
CREATE VIEW indicator_broker_ev_status AS
WITH version_returns AS (
  SELECT symbol, indicator_period_seconds, indicator_model_version,
         count(*) FILTER (WHERE quote_cohort='confirmatory' AND quote_ev > 0
           AND quote_price > 0 AND quote_payout > quote_price AND abs(quote_price - buy_price) <= 0.01
           AND quote_ev >= 0.01 * quote_payout / quote_price - 1e-9) AS purchased,
         count(*) FILTER (WHERE quote_cohort='exploratory') AS exploratory_purchased,
         count(*) FILTER (WHERE quote_cohort='confirmatory' AND quote_ev > 0
           AND quote_price > 0 AND quote_payout > quote_price AND abs(quote_price - buy_price) <= 0.01
           AND quote_ev >= 0.01 * quote_payout / quote_price - 1e-9 AND settlement_source='broker'
           AND profit IS NOT NULL) AS settled,
         avg(profit / buy_price) FILTER (WHERE quote_cohort='confirmatory' AND quote_ev > 0
           AND quote_price > 0 AND quote_payout > quote_price AND abs(quote_price - buy_price) <= 0.01
           AND quote_ev >= 0.01 * quote_payout / quote_price - 1e-9
           AND settlement_source='broker' AND profit IS NOT NULL) AS mean_unit_return,
         stddev_samp(profit / buy_price) FILTER (WHERE quote_cohort='confirmatory' AND quote_ev > 0
           AND quote_price > 0 AND quote_payout > quote_price AND abs(quote_price - buy_price) <= 0.01
           AND quote_ev >= 0.01 * quote_payout / quote_price - 1e-9
           AND settlement_source='broker' AND profit IS NOT NULL) AS unit_return_sd,
         avg(quote_ev) FILTER (WHERE quote_cohort='confirmatory' AND quote_ev > 0
           AND quote_price > 0 AND quote_payout > quote_price AND abs(quote_price - buy_price) <= 0.01
           AND quote_ev >= 0.01 * quote_payout / quote_price - 1e-9
           AND settlement_source='broker' AND profit IS NOT NULL) AS mean_quoted_ev
  FROM contract_executions
  WHERE account_mode='demo' AND indicator_model_version IS NOT NULL
    AND indicator_period_seconds IN (300,900,3600)
    AND buy_price > 0 AND coalesce(contract_type, direction) IN ('CALL','PUT')
  GROUP BY symbol, indicator_period_seconds, indicator_model_version
)
SELECT symbol, indicator_period_seconds, indicator_model_version, purchased, exploratory_purchased,
       settled, mean_unit_return, mean_quoted_ev,
       CASE WHEN settled >= 2 THEN
         mean_unit_return - 1.653 * unit_return_sd / sqrt(settled)
       END AS lower_95,
       CASE WHEN settled >= 200 THEN
         mean_unit_return - 1.653 * unit_return_sd / sqrt(settled) > 0
       ELSE false END AS promoted
FROM version_returns;
