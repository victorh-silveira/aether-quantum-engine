"""Configuracao de fixtures e paths para testes do indicador."""

import sys
from pathlib import Path


_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
