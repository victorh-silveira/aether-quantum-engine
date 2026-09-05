CREATE OR REPLACE FUNCTION aether_apply_hypertable_lifecycle(
  p_hypertable text,
  p_segmentby text,
  p_compress_after interval,
  p_retention_after interval DEFAULT NULL,
  p_orderby text DEFAULT 'time DESC'
) RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  ht regclass;
BEGIN
  ht := to_regclass(p_hypertable);
  IF ht IS NULL THEN
    RAISE NOTICE 'aether lifecycle: hypertable % ausente, ignorando', p_hypertable;
    RETURN;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM timescaledb_information.hypertables h
    WHERE format('%I.%I', h.hypertable_schema, h.hypertable_name)::regclass = ht
  ) THEN
    RAISE NOTICE 'aether lifecycle: % nao e hypertable, ignorando', p_hypertable;
    RETURN;
  END IF;

  EXECUTE format(
    'ALTER TABLE %s SET (timescaledb.compress = true, timescaledb.compress_segmentby = %L, timescaledb.compress_orderby = %L)',
    ht,
    p_segmentby,
    p_orderby
  );

  PERFORM add_compression_policy(ht, p_compress_after, if_not_exists => TRUE);

  IF p_retention_after IS NOT NULL THEN
    PERFORM add_retention_policy(ht, p_retention_after, if_not_exists => TRUE);
  END IF;
END;
$$;

SELECT aether_apply_hypertable_lifecycle(
  'public.ticks',
  'symbol',
  INTERVAL '7 days',
  INTERVAL '30 days',
  'time DESC'
);

SELECT aether_apply_hypertable_lifecycle(
  'public.ohlc_bars',
  'symbol,granularity',
  INTERVAL '7 days',
  NULL,
  'time DESC, epoch DESC'
);
