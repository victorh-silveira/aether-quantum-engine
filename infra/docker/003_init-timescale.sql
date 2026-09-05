CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS ticks (
  time TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  epoch_ms BIGINT NOT NULL,
  price DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable(
  'ticks',
  'time',
  chunk_time_interval => INTERVAL '1 day',
  if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS ohlc_bars (
  time TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  epoch BIGINT NOT NULL,
  granularity INT NOT NULL,
  open DOUBLE PRECISION,
  high DOUBLE PRECISION,
  low DOUBLE PRECISION,
  close DOUBLE PRECISION,
  tick_count INT,
  mean_inter_tick_ms DOUBLE PRECISION,
  price_velocity DOUBLE PRECISION
);

SELECT create_hypertable(
  'ohlc_bars',
  'time',
  chunk_time_interval => INTERVAL '1 day',
  if_not_exists => TRUE
);

SELECT set_chunk_time_interval('ticks', INTERVAL '1 day');
SELECT set_chunk_time_interval('ohlc_bars', INTERVAL '1 day');

CREATE UNIQUE INDEX IF NOT EXISTS ohlc_bars_symbol_epoch
  ON ohlc_bars (symbol, epoch, granularity, time);

CREATE INDEX IF NOT EXISTS ohlc_bars_symbol_gran_time
  ON ohlc_bars (symbol, granularity, time DESC);

CREATE INDEX IF NOT EXISTS ticks_symbol_time
  ON ticks (symbol, time DESC);
