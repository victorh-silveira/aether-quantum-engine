"""Gerenciamento de autenticação para o Aether Engine."""

import logging
import os

from dotenv import load_dotenv


class AuthManager:
    """Gerencia tokens de autenticação e fluxos OAuth2 para o Aether Engine.

    Este serviço lida com a recuperação de tokens de variáveis de ambiente e
    pode ser estendido para gerenciar handshakes OAuth2/PKCE completos.
    """

    def __init__(self, mode: str = "demo"):
        """Inicializa o AuthManager com o modo de negociação especificado.

        Args:
            mode (str): O modo de negociação, 'demo' ou 'live'.
        """
        load_dotenv()
        self.mode = mode
        env_suffix = "LIVE" if mode.lower() in ["live", "real"] else mode.upper()
        self.token = os.getenv(f"AETHER_{env_suffix}_TOKEN")
        self.logger = logging.getLogger("AETH")
        self.logger.debug("AUTH | modo=%s", mode.upper())

    def get_token(self) -> str:
        """Recupera o token de autenticação para o modo atual.

        Returns:
            str: O token da API ou None se não for encontrado.
        """
        return self.token
