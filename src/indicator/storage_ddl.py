"""Definicao de esquemas DDL SQL e queries de promocao para persistencia do indicador."""

INIT_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS indicator_symbols (
    symbol TEXT PRIMARY KEY, name TEXT NOT NULL, active BOOLEAN NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS indicator_signals (
    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL, candle_epoch BIGINT NOT NULL,
    side TEXT NOT NULL, probability DOUBLE PRECISION, model_version TEXT,
    reason TEXT NOT NULL, close DOUBLE PRECISION NOT NULL,
    observed_side TEXT, correct BOOLEAN,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, period_seconds, candle_epoch)
);
CREATE TABLE IF NOT EXISTS indicator_model_versions (
    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    PRIMARY KEY (symbol, period_seconds)
);
CREATE TABLE IF NOT EXISTS indicator_live (
    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL,
    signal_epoch BIGINT NOT NULL, observed_at BIGINT NOT NULL,
    spot DOUBLE PRECISION NOT NULL, delta DOUBLE PRECISION NOT NULL,
    interim_side TEXT NOT NULL,
    PRIMARY KEY (symbol, period_seconds)
);
CREATE TABLE IF NOT EXISTS indicator_candles (
    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL,
    epoch BIGINT NOT NULL, time TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL, high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL, close DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, period_seconds, epoch)
);
CREATE TABLE IF NOT EXISTS indicator_model_registry (
    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL,
    model_version TEXT NOT NULL, state TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, period_seconds, model_version),
    CHECK (state IN ('pesquisa', 'piloto', 'ativo', 'reprovado'))
);
CREATE TABLE IF NOT EXISTS indicator_candidate_signals (
    symbol TEXT NOT NULL, period_seconds INTEGER NOT NULL,
    candle_epoch BIGINT NOT NULL, model_version TEXT NOT NULL,
    side TEXT NOT NULL, probability DOUBLE PRECISION,
    reason TEXT NOT NULL, close DOUBLE PRECISION NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, period_seconds, candle_epoch, model_version)
);
"""

REPROVE_FAILED_PILOTS_SQL = """
UPDATE indicator_model_registry r SET state='reprovado', updated_at=now()
FROM indicator_broker_ev_status e
WHERE r.symbol=e.symbol AND r.period_seconds=e.indicator_period_seconds
  AND r.model_version=e.indicator_model_version AND r.state='piloto'
  AND e.purchased >= 200 AND e.settled >= 200 AND NOT e.promoted
"""

DEACTIVATE_REPLACED_ACTIVES_SQL = """
UPDATE indicator_model_registry r SET state='reprovado', updated_at=now()
WHERE r.state='ativo' AND EXISTS (
    SELECT 1 FROM indicator_model_registry pilot
    JOIN indicator_broker_ev_status e ON e.symbol=pilot.symbol
      AND e.indicator_period_seconds=pilot.period_seconds
      AND e.indicator_model_version=pilot.model_version
    WHERE pilot.symbol=r.symbol AND pilot.period_seconds=r.period_seconds
      AND pilot.state='piloto' AND e.promoted
)
"""

ACTIVATE_PROMOTED_PILOTS_SQL = """
UPDATE indicator_model_registry r SET state='ativo', updated_at=now()
FROM (
    SELECT DISTINCT ON (p.symbol, p.period_seconds)
        p.symbol, p.period_seconds, p.model_version
    FROM indicator_model_registry p
    JOIN indicator_broker_ev_status e ON e.symbol=p.symbol
      AND e.indicator_period_seconds=p.period_seconds
      AND e.indicator_model_version=p.model_version
    WHERE p.state='piloto' AND e.promoted
    ORDER BY p.symbol, p.period_seconds, p.created_at ASC, p.model_version
) selected
WHERE r.symbol=selected.symbol AND r.period_seconds=selected.period_seconds
  AND r.model_version=selected.model_version AND r.state='piloto'
"""

REPROVE_EXCESS_PILOTS_SQL = """
UPDATE indicator_model_registry r SET state='reprovado', updated_at=now()
FROM indicator_broker_ev_status e
WHERE r.symbol=e.symbol AND r.period_seconds=e.indicator_period_seconds
  AND r.model_version=e.indicator_model_version AND r.state='piloto'
  AND e.promoted AND EXISTS (
      SELECT 1 FROM indicator_model_registry active
      WHERE active.symbol=r.symbol AND active.period_seconds=r.period_seconds
        AND active.state='ativo'
  )
"""
