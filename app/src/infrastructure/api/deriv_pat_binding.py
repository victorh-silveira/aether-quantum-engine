"""Resolucao e cache de App ID para PAT Deriv."""

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from src.infrastructure.api.deriv_http import read_http_response


LEGACY_DERIV_APP_IDS = frozenset({"1089", "16929", "36544"})


def looks_like_pat(value: str) -> bool:
    """Indica se o valor parece um token PAT Deriv."""
    return value.startswith("pat_")


class DerivPatBindingError(Exception):
    """App ID nao encontrado ou invalido para a PAT informada."""


def pat_fingerprint(pat: str) -> str:
    """Gera identificador estavel para cache de binding sem expor a PAT."""
    return hashlib.sha256(pat.strip().encode("utf-8")).hexdigest()[:24]


def parse_deriv_pat(value: str) -> tuple[str, str | None]:
    """Separa PAT e App ID quando o valor usa separador composto."""
    raw = value.strip()
    if not raw:
        return "", None
    for sep in ("|", "@"):
        if sep not in raw:
            continue
        left, right = raw.split(sep, 1)
        token = left.strip()
        app_id = right.strip()
        if token and app_id and not looks_like_pat(app_id):
            return token, app_id
    return raw, None


def binding_path(repo_root: Path) -> Path:
    """Caminho do arquivo JSON de bindings PAT para App ID."""
    return repo_root / "app" / "data" / "deriv" / "pat_bindings.json"


def load_binding(repo_root: Path, pat: str) -> str | None:
    """Le App ID em cache para a PAT, se existir."""
    path = binding_path(repo_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    row = data.get(pat_fingerprint(pat))
    if isinstance(row, dict):
        app_id = row.get("app_id")
        if isinstance(app_id, str) and app_id.strip():
            return app_id.strip()
    return None


def save_binding(repo_root: Path, pat: str, app_id: str) -> None:
    """Persiste App ID validado para a PAT no cache local."""
    path = binding_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}
    data[pat_fingerprint(pat)] = {"app_id": app_id.strip()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_config_app_id(repo_root: Path) -> str:
    """Le App ID de arquivos de configuracao no repositorio."""
    for rel in ("config/deriv_pat_app_id", "config/deriv_pat_app_id.txt"):
        path = repo_root / rel
        if not path.is_file():
            continue
        line = path.read_text(encoding="utf-8").strip().splitlines()
        if line and line[0].strip() and not line[0].strip().startswith("#"):
            return line[0].strip()
    return ""


def read_candidate_app_ids(repo_root: Path) -> list[str]:
    """Lista candidatos a App ID para descoberta por sonda REST."""
    out: list[str] = []
    env_list = os.getenv("AETHER_DERIV_APP_ID_CANDIDATES", "")
    if env_list.strip():
        out.extend(x.strip() for x in env_list.split(",") if x.strip())
    path = repo_root / "config" / "deriv_pat_app_id.candidates"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            val = line.strip()
            if val and not val.startswith("#"):
                out.append(val)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in out:
        if item not in seen and item not in LEGACY_DERIV_APP_IDS:
            seen.add(item)
            ordered.append(item)
    return ordered


def probe_accounts_ok(pat: str, app_id: str, rest_base: str, timeout: float) -> bool:
    """Testa se PAT e App ID aceitam GET /options/accounts."""
    url = f"{rest_base.rstrip('/')}/trading/v1/options/accounts"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Deriv-App-ID": app_id,
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        read_http_response(req, timeout)
        return True
    except urllib.error.HTTPError:
        return False
    except urllib.error.URLError:
        return False


def discover_app_id_for_pat(
    pat: str,
    repo_root: Path,
    *,
    rest_base: str = "https://api.derivws.com",
    timeout: float = 30.0,
    explicit: str | None = None,
) -> str:
    """Resolve App ID por env, cache, config ou sonda em candidatos."""
    if explicit and explicit.strip():
        return explicit.strip()
    _, from_pat = parse_deriv_pat(pat)
    if from_pat:
        return from_pat
    cached = load_binding(repo_root, pat)
    if cached:
        return cached
    cfg = read_config_app_id(repo_root)
    if cfg:
        return cfg
    for key in ("AETHER_DERIV_APP_ID", "DERIV_APP_ID"):
        val = os.getenv(key)
        if val and val.strip():
            return val.strip()
    for candidate in read_candidate_app_ids(repo_root):
        if probe_accounts_ok(pat, candidate, rest_base, timeout):
            save_binding(repo_root, pat, candidate)
            return candidate
    raise DerivPatBindingError(
        "App ID nao encontrado para esta PAT. A Deriv exige PAT + App ID do mesmo app em developers.deriv.com. "
        "Opcoes organicas: (1) AETHER_DERIV_PAT=pat_...|SEU_APP_ID no .env; "
        "(2) arquivo config/deriv_pat_app_id com uma linha; "
        "(3) python app/scripts/deriv_pat_connect.py --app-id SEU_ID --save-binding"
    )
