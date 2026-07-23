"""Gerenciador de persistência para salvar e carregar o estado do trading."""

import contextlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from aether_paths import repo_path


_PERMISSION_OS_ERRORS = (PermissionError, OSError)


class PersistenceManager:
    """Lida com a serialização e persistência do estado do motor de trading.

    Este gerenciador garante que os dados críticos da sessão sejam salvos em disco
    e possam ser recuperados após um reinício ou falha do sistema.
    """

    def __init__(self, file_path: str | Path | None = None):
        self.file_path = Path(file_path) if file_path is not None else repo_path("data", "state.json")
        self.logger = logging.getLogger("AETH")
        self._save_lock = threading.Lock()
        self._ensure_directory()
        self._prune_stale_temp_files()

    def _ensure_directory(self):
        """Cria o diretório de dados se ele não existir."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _unique_temp_path(self) -> Path:
        """Gera caminho temporario unico para evitar colisao entre saves concorrentes."""
        token = uuid.uuid4().hex
        return self.file_path.with_name(f".state.{os.getpid()}.{token}.tmp")

    def _prune_stale_temp_files(self) -> None:
        """Remove arquivos temporarios antigos deixados por falhas anteriores."""
        parent = self.file_path.parent
        if not parent.is_dir():
            return
        for candidate in parent.glob(".state.*.tmp"):
            with contextlib.suppress(Exception):
                candidate.unlink()

    def _write_temp(self, data: dict[str, Any], temp_file: Path) -> None:
        """Grava JSON no temporario com flush em disco."""
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def _publish_temp(self, temp_file: Path) -> None:
        """Publica temporario no destino com retry e fallback Windows-safe."""
        for i in range(8):
            try:
                temp_file.replace(self.file_path)
                return
            except FileNotFoundError:
                if not temp_file.exists():
                    raise
                time.sleep(0.05 * (i + 1))
            except _PERMISSION_OS_ERRORS:
                if i == 7:
                    break
                time.sleep(0.05 * (i + 1))
        if temp_file.exists():
            shutil.copy2(temp_file, self.file_path)
            with contextlib.suppress(Exception):
                temp_file.unlink()

    def save(self, data: dict[str, Any]):
        """Salva os dados em um arquivo JSON de forma atômica com verificação de diretório.

        Args:
            data (Dict[str, Any]): Os dados de estado a serem persistidos.
        """
        with self._save_lock:
            self._ensure_directory()
            temp_file = self._unique_temp_path()
            try:
                self._write_temp(data, temp_file)
                self._publish_temp(temp_file)
            except Exception as e:
                self.logger.error(f"PERS: Erro critico ao salvar estado: {e}")
                with contextlib.suppress(Exception):
                    if temp_file.exists():
                        temp_file.unlink()
            finally:
                if temp_file.exists():
                    with contextlib.suppress(Exception):
                        temp_file.unlink()

    def load(self) -> dict[str, Any] | None:
        """Carrega os dados de estado do arquivo JSON.

        Returns:
            Dict[str, Any] | None: Os dados carregados ou None se o arquivo não existir ou for inválido.
        """
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            return None

        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"PERS: Arquivo de estado corrompido, iniciando novo: {e}")
            return None
        except Exception as e:
            self.logger.error(f"PERS: Erro inesperado ao carregar estado: {e}")
            return None
