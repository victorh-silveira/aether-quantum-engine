"""Testes de configuracao Redis Docker."""

from pathlib import Path


def test_redis_conf_uses_aof_everysec():
    root = Path(__file__).resolve().parents[4]
    conf = root / "infra" / "docker" / "config" / "redis.conf"
    if not conf.is_file():
        conf = root / "infra" / "docker" / "redis.conf"
    text = conf.read_text(encoding="utf-8")
    assert "appendonly yes" in text
    assert "appendfsync everysec" in text
    assert 'save ""' in text
    assert "maxmemory 200mb" in text
    assert "maxmemory-policy noeviction" in text
    assert "io-threads 2" in text
    assert "io-threads-do-reads yes" in text


def test_settings_redis_prefers_loopback_ipv4():
    settings = Path(__file__).resolve().parents[4] / "config" / "settings.json"
    import json

    payload = json.loads(settings.read_text(encoding="utf-8"))
    redis_cfg = payload["infra"]["redis"]
    assert "127.0.0.1" in str(redis_cfg["url"])
    assert float(redis_cfg["socket_connect_timeout_seconds"]) > 0.0
    assert float(redis_cfg["socket_timeout_seconds"]) > 0.0


def test_host_prereq_script_exists():
    root = Path(__file__).resolve().parents[4]
    script = root / "infra" / "docker" / "sh" / "host-prereq.sh"
    if not script.is_file():
        script = root / "infra" / "docker" / "host-prereq.sh"
    assert script.is_file()
    content = script.read_text(encoding="utf-8")
    assert "vm.overcommit_memory=1" in content
