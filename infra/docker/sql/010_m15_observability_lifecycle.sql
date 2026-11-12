DO $$
BEGIN
  IF to_regclass('public.contract_executions') IS NOT NULL THEN
    ALTER TABLE contract_executions ADD COLUMN IF NOT EXISTS trace_id TEXT;
    CREATE INDEX IF NOT EXISTS idx_contract_executions_symbol_period_source_start
      ON contract_executions (symbol, indicator_period_seconds, settlement_source, date_start DESC);
    CREATE INDEX IF NOT EXISTS idx_contract_executions_trace_id
      ON contract_executions (trace_id) WHERE trace_id IS NOT NULL;
  END IF;

  IF to_regclass('public.pullback_signals') IS NOT NULL THEN
    CREATE INDEX IF NOT EXISTS idx_pullback_signals_sym_per_ep
      ON pullback_signals (symbol, period_seconds, candle_epoch DESC);
  END IF;

  IF to_regclass('public.ticks') IS NOT NULL AND NOT EXISTS (
    SELECT 1
    FROM timescaledb_information.continuous_aggregates
    WHERE view_name = 'candle_m15'
  ) THEN
    EXECUTE $cag$
      CREATE MATERIALIZED VIEW candle_m15
      WITH (timescaledb.continuous) AS
      SELECT
        time_bucket(INTERVAL '15 minutes', time) AS bucket,
        symbol,
        first(price, time) AS open,
        max(price) AS high,
        min(price) AS low,
        last(price, time) AS close,
        count(*)::bigint AS tick_count
      FROM ticks
      GROUP BY 1, 2
      WITH NO DATA
    $cag$;
  END IF;

  IF to_regclass('public.candle_m15') IS NOT NULL THEN
    PERFORM add_continuous_aggregate_policy(
      'candle_m15',
      start_offset => INTERVAL '2 days',
      end_offset => INTERVAL '15 minutes',
      schedule_interval => INTERVAL '15 minutes',
      if_not_exists => TRUE
    );
  END IF;

  IF to_regclass('public.contract_executions') IS NOT NULL AND to_regclass('public.ohlc_bars') IS NOT NULL THEN
    CREATE OR REPLACE VIEW contract_label_audit_m15 AS
    SELECT
      c.contract_id,
      c.symbol,
      c.account_mode,
      c.contract_type,
      c.date_start,
      c.date_expiry,
      c.buy_price,
      c.payout,
      round(((c.payout - c.buy_price) / NULLIF(c.buy_price, 0))::numeric, 6) AS observed_payout_rate,
      round((c.buy_price / NULLIF(c.payout, 0))::numeric, 6) AS observed_breakeven_prob,
      c.entry_tick,
      c.exit_tick,
      (c.exit_tick - c.entry_tick) AS broker_spot_delta,
      c.profit,
      c.status,
      c.settlement_source,
      c.trace_id,
      c.indicator_candle_epoch,
      b.epoch AS bar_epoch,
      (c.date_start % 900) AS entry_phase_seconds,
      (c.date_expiry - c.date_start) AS observed_duration_seconds,
      b.open AS m15_open,
      b.close AS m15_close,
      (b.close - b.open) AS m15_spot_delta,
      CASE
        WHEN b.close > b.open THEN 'CALL'
        WHEN b.close < b.open THEN 'PUT'
        ELSE 'TIE'
      END AS m15_expected_direction,
      CASE
        WHEN c.exit_tick > c.entry_tick THEN 'CALL'
        WHEN c.exit_tick < c.entry_tick THEN 'PUT'
        ELSE 'TIE'
      END AS broker_observed_outcome
    FROM contract_executions c
    LEFT JOIN LATERAL (
      SELECT ob.epoch, ob.open, ob.close
      FROM ohlc_bars ob
      WHERE ob.symbol = c.symbol
        AND ob.granularity = 900
        AND ob.epoch BETWEEN (c.date_start - 180) AND (c.date_start + 180)
      ORDER BY abs(ob.epoch - c.date_start) ASC
      LIMIT 1
    ) b ON TRUE
    WHERE c.contract_type IN ('CALL', 'PUT')
      AND c.settlement_source IN ('broker', 'profit_table', 'inferred_rest')
      AND (c.indicator_period_seconds = 900 OR (c.date_expiry - c.date_start) BETWEEN 895 AND 905);
  END IF;
END
$$;
