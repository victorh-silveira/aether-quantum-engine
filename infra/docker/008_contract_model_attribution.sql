ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS model_version TEXT;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS calibrated_call_prob DOUBLE PRECISION;

CREATE OR REPLACE VIEW contract_model_outcomes AS
SELECT model_version, symbol, account_mode, contract_type,
       count(*) AS broker_settled_count,
       count(*) FILTER (WHERE profit > 0) AS wins,
       avg(profit / buy_price) AS mean_unit_return,
       avg(power(calibrated_call_prob - (exit_tick > entry_tick)::int, 2)) AS calibrated_brier
FROM contract_executions
WHERE model_version IS NOT NULL
  AND calibrated_call_prob BETWEEN 0 AND 1
  AND settlement_source = 'broker'
  AND contract_type IN ('CALL', 'PUT')
  AND entry_tick IS NOT NULL AND exit_tick IS NOT NULL
  AND buy_price > 0 AND profit IS NOT NULL
GROUP BY model_version, symbol, account_mode, contract_type;
