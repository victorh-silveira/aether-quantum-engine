"""HTTP seguro (esquemas http/https) para chamadas Deriv."""

import urllib.parse
import urllib.request


_ALLOWED_SCHEMES = frozenset({"https", "http"})


def read_http_response(req: urllib.request.Request, timeout: float) -> bytes:
    """Abre URL http/https e retorna o corpo da resposta."""
    scheme = urllib.parse.urlparse(req.full_url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        msg = f"Esquema de URL nao permitido: {scheme}"
        raise ValueError(msg)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()
