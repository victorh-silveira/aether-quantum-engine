"""Testes unitários para o PersistenceManager."""

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
    temp_path = tmp_path / "fail.tmp"
    manager = PersistenceManager(file_path=str(file_path))

    temp_path.touch()

    def mock_dump(*args, **kwargs):
        raise OSError("Relay error")

    monkeypatch.setattr("json.dump", mock_dump)

    manager.save({"any": "data"})
    assert not temp_path.exists()


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

    def mock_rename(self, target):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise PermissionError("Locked")

    monkeypatch.setattr("pathlib.Path.rename", mock_rename)
    monkeypatch.setattr("time.sleep", lambda x: None)

    temp_persistence.save({"retry": "ok"})
    assert call_count == 3


def test_persistence_save_retry_failure(temp_persistence, monkeypatch):
    """Verifica se o salvamento falha após 5 tentativas de PermissionError."""

    def mock_rename(self, target):
        raise PermissionError("Still locked")

    monkeypatch.setattr("pathlib.Path.rename", mock_rename)
    monkeypatch.setattr("time.sleep", lambda x: None)

    temp_persistence.save({"retry": "fail"})


def test_persistence_save_unlink_error_suppressed(temp_persistence, monkeypatch):
    """Verifica se a falha ao remover o arquivo temporário durante a recuperação é suprimida."""

    def mock_dump(*args, **kwargs):
        raise OSError("Save fail")

    def mock_unlink(self):
        raise OSError("Unlink fail")

    monkeypatch.setattr("json.dump", mock_dump)
    monkeypatch.setattr("pathlib.Path.unlink", mock_unlink)

    temp_persistence.save({"any": "data"})
