ALTER SYSTEM SET checkpoint_completion_target = 0.95;
ALTER SYSTEM SET max_wal_size = '2GB';
SELECT pg_reload_conf();
