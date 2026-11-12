"""Queries SQL e mapeamento de campos de contratos para TimescaleMarketWriter."""

CONTRACT_FIELDS = (
    "contract_id",
    "symbol",
    "account_mode",
    "direction",
    "transaction_buy_id",
    "request_epoch_ms",
    "ack_epoch_ms",
    "date_start",
    "date_expiry",
    "entry_tick",
    "entry_tick_time",
    "exit_tick",
    "exit_tick_time",
    "buy_price",
    "payout",
    "signal_prob",
    "profit",
    "status",
    "settlement_source",
    "proposal_id",
    "contract_type",
    "barrier",
    "model_version",
    "calibrated_call_prob",
    "indicator_model_version",
    "indicator_call_prob",
    "indicator_candle_epoch",
    "quote_price",
    "quote_payout",
    "quote_ev",
    "indicator_period_seconds",
    "quote_cohort",
    "trace_id",
)

CONTRACT_UPSERT_SQL = (
    "INSERT INTO contract_executions (contract_id, symbol, account_mode, direction, transaction_buy_id, "
    "request_epoch_ms, ack_epoch_ms, date_start, date_expiry, entry_tick, entry_tick_time, exit_tick, "
    "exit_tick_time, buy_price, payout, signal_prob, profit, status, settlement_source, proposal_id, "
    "contract_type, barrier, model_version, calibrated_call_prob, indicator_model_version, "
    "indicator_call_prob, indicator_candle_epoch, quote_price, quote_payout, quote_ev, "
    "indicator_period_seconds, quote_cohort, trace_id) "
    "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33) "
    "ON CONFLICT (contract_id) DO UPDATE SET "
    "proposal_id=COALESCE(EXCLUDED.proposal_id, contract_executions.proposal_id), "
    "contract_type=COALESCE(EXCLUDED.contract_type, contract_executions.contract_type), "
    "barrier=COALESCE(EXCLUDED.barrier, contract_executions.barrier), "
    "model_version=COALESCE(EXCLUDED.model_version, contract_executions.model_version), "
    "calibrated_call_prob=COALESCE(EXCLUDED.calibrated_call_prob, contract_executions.calibrated_call_prob), "
    "indicator_model_version=COALESCE(EXCLUDED.indicator_model_version, contract_executions.indicator_model_version), "
    "indicator_call_prob=COALESCE(EXCLUDED.indicator_call_prob, contract_executions.indicator_call_prob), "
    "indicator_candle_epoch=COALESCE(EXCLUDED.indicator_candle_epoch, contract_executions.indicator_candle_epoch), "
    "quote_price=COALESCE(EXCLUDED.quote_price, contract_executions.quote_price), "
    "quote_payout=COALESCE(EXCLUDED.quote_payout, contract_executions.quote_payout), "
    "quote_ev=COALESCE(EXCLUDED.quote_ev, contract_executions.quote_ev), "
    "indicator_period_seconds=COALESCE(EXCLUDED.indicator_period_seconds, contract_executions.indicator_period_seconds), "
    "quote_cohort=COALESCE(EXCLUDED.quote_cohort, contract_executions.quote_cohort), "
    "trace_id=COALESCE(EXCLUDED.trace_id, contract_executions.trace_id), "
    "transaction_buy_id=COALESCE(EXCLUDED.transaction_buy_id, contract_executions.transaction_buy_id), "
    "request_epoch_ms=COALESCE(EXCLUDED.request_epoch_ms, contract_executions.request_epoch_ms), "
    "ack_epoch_ms=COALESCE(EXCLUDED.ack_epoch_ms, contract_executions.ack_epoch_ms), "
    "date_start=COALESCE(EXCLUDED.date_start, contract_executions.date_start), "
    "date_expiry=COALESCE(EXCLUDED.date_expiry, contract_executions.date_expiry), "
    "entry_tick=COALESCE(EXCLUDED.entry_tick, contract_executions.entry_tick), "
    "entry_tick_time=COALESCE(EXCLUDED.entry_tick_time, contract_executions.entry_tick_time), "
    "exit_tick=COALESCE(EXCLUDED.exit_tick, contract_executions.exit_tick), "
    "exit_tick_time=COALESCE(EXCLUDED.exit_tick_time, contract_executions.exit_tick_time), "
    "buy_price=COALESCE(EXCLUDED.buy_price, contract_executions.buy_price), "
    "payout=COALESCE(EXCLUDED.payout, contract_executions.payout), "
    "signal_prob=COALESCE(EXCLUDED.signal_prob, contract_executions.signal_prob), "
    "profit=CASE WHEN EXCLUDED.settlement_source IN ('broker', 'profit_table') OR contract_executions.profit IS NULL "
    "THEN EXCLUDED.profit ELSE contract_executions.profit END, "
    "status=CASE WHEN EXCLUDED.settlement_source IN ('broker', 'profit_table') OR contract_executions.status IS NULL "
    "THEN EXCLUDED.status ELSE contract_executions.status END, "
    "settlement_source=CASE WHEN EXCLUDED.settlement_source='broker' THEN 'broker' "
    "WHEN contract_executions.settlement_source='broker' THEN 'broker' "
    "WHEN EXCLUDED.settlement_source='profit_table' THEN 'profit_table' "
    "WHEN contract_executions.settlement_source='profit_table' THEN 'profit_table' "
    "ELSE EXCLUDED.settlement_source END, updated_at=now()"
)
