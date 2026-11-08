"""Normalização de lista de símbolos e âncora a partir do JSON de configuração."""

import json
import re
from pathlib import Path


_VOLATILITY_SYMBOL = re.compile(r"(?:R_[0-9]+|1HZ[0-9]+V|RDBEAR|RDBULL)\Z")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def approved_indicator_symbols(models_dir: Path) -> list[str]:
    """Descobre indices M5 aprovados no treino publico do indicador."""
    result = []
    for path in sorted(models_dir.glob("*_300.json")):
        symbol = path.name.removesuffix("_300.json")
        if not _VOLATILITY_SYMBOL.fullmatch(symbol):
            continue
        try:
            model = json.loads(path.read_text(encoding="utf-8"))
            approved = (
                model.get("symbol") == symbol
                and model.get("period") == 300
                and model.get("status") == "aprovado"
                and float(model["validation_accuracy"]) > float(model["baseline_accuracy"])
                and float(model["validation_brier"]) < float(model["baseline_brier"])
            )
        except (OSError, ValueError, TypeError, KeyError):
            continue
        if approved:
            result.append(symbol)
    return result


def normalize_symbols_and_anchor(config: dict) -> tuple[str, list[str]]:
    """Deriva âncora e símbolos ativos (symbols menos excluded_symbols)."""
    anchor = str(config.get("anchor", "1HZ75V"))
    strategy = config.get("strategy", {})
    pb_cfg = config.get("pullback_strategy", {})
    raw = list(config.get("symbols") or [anchor])
    if pb_cfg.get("enabled") and (pb_cfg.get("dynamic_rotation") or pb_cfg.get("operate_all_signals")):
        rotation_syms = [
            "OTC_SSMI",
            "OTC_SPC",
            "OTC_NDX",
            "OTC_DJI",
            "OTC_SX5E",
            "OTC_GDAXI",
            "OTC_FCHI",
            "OTC_AEX",
            "OTC_HSI",
            "OTC_AS51",
            "OTC_FTSE",
            "OTC_N225",
        ]
        raw.extend([s for s in rotation_syms if s not in raw])
    indicator_cfg = config.get("indicator_consensus", {})
    if indicator_cfg.get("enabled") and indicator_cfg.get("auto_symbols") and not pb_cfg.get("enabled", False):
        models_dir = _REPOSITORY_ROOT / "indicator/data/indicator-models"
        raw = [*raw, *approved_indicator_symbols(models_dir)]
    excluded = {str(x) for x in ((strategy or {}).get("excluded_symbols") or [])}
    excluded |= {str(x) for x in (pb_cfg.get("excluded_symbols") or [])}
    symbols = [s for s in dict.fromkeys(raw) if s not in excluded or s == anchor]
    if anchor not in symbols:
        symbols.insert(0, anchor)
    return anchor, symbols


def resolve_dl_train_symbols(config: dict) -> list[str]:
    """Retorna simbolos que o treino DL pode persistir; padrao e symbols do motor."""
    dl_cfg = config.get("deep_learning") if isinstance(config.get("deep_learning"), dict) else {}
    raw = dl_cfg.get("train_symbols")
    indicator_cfg = config.get("indicator_consensus", {})
    if indicator_cfg.get("enabled") and indicator_cfg.get("auto_symbols"):
        _, symbols = normalize_symbols_and_anchor(config)
        if isinstance(raw, list):
            return list(dict.fromkeys([*(str(symbol) for symbol in raw), *symbols]))
        return symbols
    if isinstance(raw, list):
        return [str(symbol) for symbol in dict.fromkeys(raw)]
    _, symbols = normalize_symbols_and_anchor(config)
    return symbols
