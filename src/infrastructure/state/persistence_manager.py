"""Gerenciador de persistência para salvar e carregar o estado do trading."""

import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any


_PERMISSION_OS_ERRORS = (PermissionError, OSError)


class PersistenceManager:
    """Lida com a serialização e persistência do estado do motor de trading.

    Este gerenciador garante que os dados críticos da sessão sejam salvos em disco
    e possam ser recuperados após um reinício ou falha do sistema.
    """

    def __init__(self, file_path: str = "data/state.json"):
        """Inicializa o gerenciador com o caminho de armazenamento de destino.

        Args:
            file_path (str): Caminho para o arquivo de estado JSON.
        """
        self.file_path = Path(file_path)
        self.logger = logging.getLogger("AETH")
        self._ensure_directory()

    def _ensure_directory(self):
        """Cria o diretório de dados se ele não existir."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict[str, Any]):
        """Salva os dados em um arquivo JSON de forma atômica com verificação de diretório.

        Args:
            data (Dict[str, Any]): Os dados de estado a serem persistidos.
        """
        self._ensure_directory()
        temp_file = self.file_path.with_suffix(".tmp")
        try:
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            for i in range(5):
                try:
                    if self.file_path.exists():
                        self.file_path.unlink()
                    temp_file.rename(self.file_path)
                    break
                except _PERMISSION_OS_ERRORS:
                    if i == 4:
                        raise
                    time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"PERS: Erro critico ao salvar estado: {e}")
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

    def clear(self):
        """Exclui o arquivo de estado persistido."""
        if self.file_path.exists():
            self.file_path.unlink()
            self.logger.info("PERS: Estado persistido removido com sucesso.")
