"""Fixtures de macro para testes que exigem execucao LLM aprovada."""

RELAXED_MACRO_CFG = {
    "confluence_conviction_floor": 0.0,
    "divergence_min_leader_strength": 0.0,
    "divergence_min_strength_gap": 0.0,
    "indefinido_min_leader_strength": 0.0,
    "indefinido_min_strength_gap": 0.0,
    "assert_min_hmm_prob": 0.0,
    "divergence_max_conviction": 0.99,
    "statarb_z_threshold": 2.5,
}


def merge_orch_config(cfg: dict) -> dict:
    out = dict(cfg)
    strategy = dict(out.get("strategy", {}))
    macro = {**RELAXED_MACRO_CFG, **(strategy.get("macro") if isinstance(strategy.get("macro"), dict) else {})}
    strategy["macro"] = macro
    out["strategy"] = strategy
    return out
