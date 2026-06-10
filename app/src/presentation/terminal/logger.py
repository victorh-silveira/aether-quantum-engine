"""Configuração de log de terminal para o Aether Engine."""

import logging
import sys
from pathlib import Path


class _FlushStreamHandler(logging.StreamHandler):
    """Handler de stream que faz flush apos cada registro (melhor para terminal)."""

    def emit(self, record):
        """Envia o registro ao stream e chama flush."""
        super().emit(record)
        self.flush()


class BlankLineSquasher(logging.Filter):
    """Filtro que descarta linhas em branco consecutivas ou no inicio da sessao."""

    def __init__(self) -> None:
        """Inicializa o filtro considerando o inicio da sessao como linha em branco."""
        super().__init__()
        self._last_blank = True

    def filter(self, record: logging.LogRecord) -> bool:
        """Permite linha em branco apenas apos uma linha com conteudo."""
        blank = not str(record.getMessage() or "").strip()
        if blank and self._last_blank:
            return False
        self._last_blank = blank
        return True


class AetherFormatter(logging.Formatter):
    """Formatador personalizado para impor níveis de 4 letras."""

    LEVEL_MAP = {"DEBUG": "DEBG ", "INFO": "INFO ", "WARNING": "AVISO", "ERROR": "ERRO ", "CRITICAL": "CRIT "}

    def format(self, record):
        """Formata o registro de log com nomes de níveis truncados.

        Args:
            record (logging.LogRecord): O registro de log a ser formatado.

        Returns:
            str: A string de log formatada.
        """
        if not str(record.getMessage() or "").strip():
            return ""
        record.levelname = self.LEVEL_MAP.get(record.levelname, record.levelname[:4])
        return super().format(record)


def setup_logger(name: str, log_file: str = None):
    """Configura um logger com níveis de 4 letras e cabeçalhos de data.

    Args:
        name (str): O nome da instância do logger.
        log_file (str, opcional): Caminho para o arquivo de log para persistência.

    Returns:
        logging.Logger: Uma instância de logger configurada.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addFilter(BlankLineSquasher())

    formatter = AetherFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    stdout_handler = _FlushStreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
