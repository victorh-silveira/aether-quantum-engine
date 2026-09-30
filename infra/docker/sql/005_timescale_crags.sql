DO $$
BEGIN
  IF to_regclass('public.ticks') IS NULL THEN
    RAISE NOTICE 'aether crags: ticks ausente, ignorando';
    RETURN;
  END IF;

  PERFORM set_chunk_time_interval('ticks', INTERVAL '1 day');

  IF to_regclass('public.ohlc_bars') IS NOT NULL THEN
    PERFORM set_chunk_time_interval('ohlc_bars', INTERVAL '1 day');
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM timescaledb_information.continuous_aggregates
    WHERE view_name = 'candle_m5'
  ) THEN
    EXECUTE $cag$
      CREATE MATERIALIZED VIEW candle_m5
      WITH (timescaledb.continuous) AS
      SELECT
        time_bucket(INTERVAL '5 minutes', time) AS bucket,
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

  IF to_regclass('public.candle_m5') IS NOT NULL THEN
    PERFORM add_continuous_aggregate_policy(
      'candle_m5',
      start_offset => INTERVAL '1 day',
      end_offset => INTERVAL '5 minutes',
      schedule_interval => INTERVAL '5 minutes',
      if_not_exists => TRUE
    );
  END IF;
END
$$;
