ALTER SYSTEM SET checkpoint_completion_target = 0.95;
ALTER SYSTEM SET max_wal_size = '2GB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET effective_cache_size = '768MB';
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET max_parallel_workers_per_gather = 2;
SELECT pg_reload_conf();
