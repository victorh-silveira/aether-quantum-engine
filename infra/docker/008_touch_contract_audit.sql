ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS proposal_id TEXT;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS contract_type TEXT;
ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS barrier DOUBLE PRECISION;

CREATE OR REPLACE VIEW touch_contract_audit AS
SELECT contract_id, symbol, account_mode, proposal_id, contract_type, barrier,
       date_start, date_expiry, entry_tick, entry_tick_time, exit_tick, exit_tick_time,
       buy_price, payout, profit, status, settlement_source,
       profit / NULLIF(buy_price, 0) AS realized_return
FROM contract_executions
WHERE contract_type IN ('ONETOUCH', 'NOTOUCH') AND settlement_source = 'broker';
