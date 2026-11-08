"""Extracao tabular de microestrutura de ticks via Polars."""

from __future__ import annotations

import polars as pl


def _ensure_temporal_column(df: pl.DataFrame, col_name: str) -> tuple[pl.DataFrame, str]:
    """Garante que a coluna temporal esteja formatada para group_by_dynamic."""
    if df.schema[col_name].is_numeric():
        return (
            df.with_columns(pl.from_epoch(pl.col(col_name).cast(pl.Int64), time_unit="s").alias("_time_dt")),
            "_time_dt",
        )
    return df, col_name


def compute_tick_microstructure(
    df_ticks: pl.DataFrame,
    *,
    time_column: str = "timestamp",
    price_column: str = "price",
    window_seconds: int = 300,
) -> pl.DataFrame:
    """Extrai assimetria de fluxo, aceleracao e pressao intrabarra agregando ticks."""
    if df_ticks.is_empty():
        return pl.DataFrame(
            schema={
                time_column: pl.Datetime("ms"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "buy_tick_ratio": pl.Float64,
                "return_autocorr": pl.Float64,
                "final_momentum": pl.Float64,
                "realized_volatility": pl.Float64,
            }
        )

    prepared_df, active_time_col = _ensure_temporal_column(df_ticks, time_column)
    tail_n = max(2, int(window_seconds * 0.15))

    transformed = (
        prepared_df.sort(active_time_col)
        .with_columns(
            [
                (pl.col(price_column) - pl.col(price_column).shift(1)).alias("_tick_delta"),
                (pl.col(price_column) / pl.col(price_column).shift(1)).log().alias("_log_ret"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("_tick_delta") > 0)
                .then(1)
                .when(pl.col("_tick_delta") < 0)
                .then(-1)
                .otherwise(0)
                .alias("_tick_sign")
            ]
        )
    )

    aggregated = transformed.group_by_dynamic(
        active_time_col,
        every=f"{int(window_seconds)}s",
    ).agg(
        [
            pl.col(price_column).first().alias("open"),
            pl.col(price_column).max().alias("high"),
            pl.col(price_column).min().alias("low"),
            pl.col(price_column).last().alias("close"),
            (
                pl.when(pl.col("_tick_sign").filter(pl.col("_tick_sign") != 0).count() > 0)
                .then(
                    pl.col("_tick_sign").filter(pl.col("_tick_sign") > 0).count()
                    / pl.col("_tick_sign").filter(pl.col("_tick_sign") != 0).count()
                )
                .otherwise(0.5)
            )
            .fill_null(0.5)
            .alias("buy_tick_ratio"),
            (pl.col("_log_ret") * pl.col("_log_ret").shift(1)).mean().fill_null(0.0).alias("return_autocorr"),
            (pl.col(price_column).tail(tail_n).last() - pl.col(price_column).tail(tail_n).first())
            .fill_null(0.0)
            .alias("final_momentum"),
            (pl.col("_log_ret").pow(2).sum()).sqrt().fill_null(0.0).alias("realized_volatility"),
        ]
    )

    if active_time_col != time_column:
        aggregated = aggregated.rename({active_time_col: time_column})

    return aggregated
