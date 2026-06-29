"""Testes de configuracao Redis Docker."""

from pathlib import Path


def test_redis_conf_uses_aof_everysec():
    conf = Path(__file__).resolve().parents[4] / "infra" / "docker" / "redis.conf"
    text = conf.read_text(encoding="utf-8")
    assert "appendonly yes" in text
    assert "appendfsync everysec" in text
    assert 'save ""' in text


def test_host_prereq_script_exists():
    script = Path(__file__).resolve().parents[4] / "infra" / "docker" / "host-prereq.sh"
    assert script.is_file()
    content = script.read_text(encoding="utf-8")
    assert "vm.overcommit_memory=1" in content
