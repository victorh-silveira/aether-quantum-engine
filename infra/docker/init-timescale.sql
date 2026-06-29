CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS ticks (
  time TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  epoch_ms BIGINT NOT NULL,
  price DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE);

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

SELECT create_hypertable('ohlc_bars', 'time', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS ohlc_bars_symbol_epoch
  ON ohlc_bars (symbol, epoch, granularity, time);
