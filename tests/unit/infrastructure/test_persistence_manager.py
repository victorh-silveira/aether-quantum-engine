"""Testes unitários para o PersistenceManager."""

from pathlib import Path

import pytest

from src.infrastructure.state.persistence_manager import PersistenceManager


@pytest.fixture
def temp_persistence(tmp_path):
    """Cria um PersistenceManager em um diretório temporário."""
    file_path = tmp_path / "state.json"
    return PersistenceManager(file_path=str(file_path))


def test_persistence_save_load(temp_persistence):
    """Verifica se os dados podem ser salvos e carregados corretamente."""
    data = {"key": "value", "number": 123}
    temp_persistence.save(data)

    temp_persistence.save({"new": "data"})

    loaded = temp_persistence.load()
    assert loaded == {"new": "data"}


def test_persistence_load_nonexistent(temp_persistence):
    """Verifica se carregar um arquivo inexistente retorna None."""
    assert temp_persistence.load() is None


def test_persistence_ensure_directory(tmp_path):
    """Verifica se os diretórios são criados automaticamente."""
    deep_path = tmp_path / "new_dir" / "subdir" / "state.json"
    manager = PersistenceManager(file_path=str(deep_path))

    data = {"ok": True}
    manager.save(data)
    assert deep_path.exists()


def test_persistence_save_error(temp_persistence, monkeypatch):
    """Verifica o tratamento de erro quando o salvamento falha."""

    def mock_open(*args, **kwargs):
        raise OSError("Permissão negada")

    monkeypatch.setattr("pathlib.Path.open", mock_open)
    temp_persistence.save({"data": 1})


def test_persistence_load_corrupted(temp_persistence):
    """Verifica o tratamento de erro ao carregar um arquivo JSON corrompido."""
    with temp_persistence.file_path.open("w") as f:
        f.write("{ invalid json")

    assert temp_persistence.load() is None


def test_persistence_save_error_unlinks_temp(tmp_path, monkeypatch):
    """Verifica se os arquivos temporários são limpos se o salvamento falhar."""
    file_path = tmp_path / "fail.json"
    manager = PersistenceManager(file_path=str(file_path))

    def mock_dump(*args, **kwargs):
        raise OSError("Relay error")

    monkeypatch.setattr("json.dump", mock_dump)

    manager.save({"any": "data"})
    leftovers = list(tmp_path.glob(".state.*.tmp"))
    assert leftovers == []


def test_persistence_load_empty(temp_persistence):
    """Verifica se carregar um arquivo vazio retorna None."""
    temp_persistence.file_path.touch()
    assert temp_persistence.load() is None


def test_persistence_load_unexpected_error(temp_persistence, monkeypatch):
    """Verifica o tratamento de erro quando ocorre um erro inesperado durante o carregamento."""
    data = {"any": "data"}
    temp_persistence.save(data)

    def mock_load(*args, **kwargs):
        raise RuntimeError("Unexpected failure")

    monkeypatch.setattr("json.load", mock_load)
    assert temp_persistence.load() is None


def test_persistence_save_retry_success(temp_persistence, monkeypatch):
    """Verifica se o salvamento tenta novamente após PermissionError e eventualmente tem sucesso."""
    call_count = 0
    real_replace = Path.replace

    def mock_replace(self, target):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise PermissionError("Locked")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", mock_replace)
    monkeypatch.setattr("time.sleep", lambda x: None)

    temp_persistence.save({"retry": "ok"})
    assert call_count == 3


def test_persistence_save_retry_failure(temp_persistence, monkeypatch):
    """Verifica se o salvamento falha após 8 tentativas de PermissionError."""

    def mock_replace(self, target):
        raise PermissionError("Still locked")

    monkeypatch.setattr(Path, "replace", mock_replace)
    monkeypatch.setattr("time.sleep", lambda x: None)

    temp_persistence.save({"retry": "fail"})


def test_persistence_save_unlink_error_suppressed(temp_persistence, monkeypatch):
    """Verifica se a falha ao remover o arquivo temporário durante a recuperação é suprimida."""

    def mock_dump(*args, **kwargs):
        raise OSError("Save fail")

    def mock_unlink(self, *, missing_ok=False):
        raise OSError("Unlink fail")

    monkeypatch.setattr("json.dump", mock_dump)
    monkeypatch.setattr("pathlib.Path.unlink", mock_unlink)

    temp_persistence.save({"any": "data"})


def test_persistence_prune_skips_when_parent_not_directory(temp_persistence, tmp_path):
    """Verifica prune quando o diretorio pai ainda nao existe como pasta."""
    missing_parent = tmp_path / "ghost_dir" / "state.json"
    temp_persistence.file_path = missing_parent
    temp_persistence._prune_stale_temp_files()


def test_persistence_prune_stale_temp_on_init(tmp_path):
    """Verifica se temporarios antigos sao removidos na inicializacao."""
    stale = tmp_path / ".state.9999.deadbeef.tmp"
    stale.write_text("{}", encoding="utf-8")
    PersistenceManager(file_path=str(tmp_path / "state.json"))
    assert not stale.exists()
