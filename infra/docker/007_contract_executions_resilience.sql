-- Migracao de resiliencia: atualiza contract_label_audit com LEFT JOIN, COALESCE e is_label_mismatched.
CREATE OR REPLACE VIEW contract_label_audit AS
SELECT c.contract_id, c.symbol, c.account_mode, c.direction,
       c.request_epoch_ms, c.ack_epoch_ms, c.date_start, c.date_expiry,
       c.entry_tick, c.entry_tick_time, c.exit_tick, c.exit_tick_time,
       c.buy_price, c.payout, c.profit, c.status,
       b0.epoch AS m5_entry_epoch, b0.close AS m5_entry_close,
       b1.epoch AS m5_exit_epoch, b1.close AS m5_exit_close,
       CASE c.direction WHEN 'CALL' THEN b1.close > b0.close
                        WHEN 'PUT' THEN b1.close < b0.close END AS m5_win,
       CASE c.direction WHEN 'CALL' THEN c.exit_tick > c.entry_tick
                        WHEN 'PUT' THEN c.exit_tick < c.entry_tick END AS spot_win,
       c.profit > 0 AS broker_win,
       c.signal_prob,
       CASE WHEN c.signal_prob BETWEEN 0 AND 1 THEN
         power(c.signal_prob - (c.exit_tick > c.entry_tick)::int, 2)
       END AS broker_brier,
       CASE
         WHEN b0.close IS NOT NULL AND b1.close IS NOT NULL AND c.exit_tick IS NOT NULL AND c.entry_tick IS NOT NULL THEN
           (CASE c.direction WHEN 'CALL' THEN b1.close > b0.close WHEN 'PUT' THEN b1.close < b0.close END) !=
           (CASE c.direction WHEN 'CALL' THEN c.exit_tick > c.entry_tick WHEN 'PUT' THEN c.exit_tick < c.entry_tick END)
         ELSE NULL
       END AS is_label_mismatched
FROM contract_executions c
LEFT JOIN LATERAL (
  SELECT epoch, close FROM ohlc_bars
  WHERE symbol=c.symbol AND granularity=300 AND epoch % 300=0
    AND epoch + 300 <= COALESCE(c.request_epoch_ms / 1000, c.date_start)
  ORDER BY epoch DESC LIMIT 1
) b0 ON true
LEFT JOIN ohlc_bars b1 ON b1.symbol=c.symbol AND b1.granularity=300
  AND b1.epoch=b0.epoch + 300 AND b1.epoch % 300=0
WHERE c.settlement_source='broker'
  AND COALESCE(c.request_epoch_ms / 1000, c.date_start) IS NOT NULL
  AND c.entry_tick IS NOT NULL AND c.exit_tick IS NOT NULL
  AND c.profit IS NOT NULL AND c.buy_price > 0;
