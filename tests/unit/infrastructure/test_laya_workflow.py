"""Fluxo GPU testado sem rede, pesos ou ordens reais."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest

from src.domain.models.market_data import Candle
from tests.unit.infrastructure.test_laya_context import context_agent
from tools.laya_workflow import artifacts, dataset, evaluation, training


def synthetic_bars(count=1120):
    """Serie alternada para verificar rotulos e separacao sem consultar o broker."""
    return [
        Candle(
            "R_10",
            100,
            102,
            98,
            99 if i % 2 else 101,
            datetime.fromtimestamp(1764000000 + i * 300, tz=UTC),
            1764000000 + i * 300,
        )
        for i in range(count)
    ]


@pytest.fixture
async def prepared(tmp_path, monkeypatch):
    """Cria as quatro particoes com a preparacao real e fonte simulada."""
    bars = synthetic_bars()

    async def fetch(path, start, end, config):
        pl.DataFrame({"epoch": [bar.epoch for bar in bars]}).write_parquet(path)
        return bars

    monkeypatch.setattr(dataset, "complete_history", fetch)
    directory = tmp_path / "data"
    await dataset.prepare(directory, {}, bars[-1].epoch + 300, days=3)
    return directory


def test_causal_states_temporal_purge_ties_and_gaps():
    bars = synthetic_bars()
    start, cutoff = bars[120].epoch, bars[-1].epoch + 300
    splits, stats = dataset.build_splits(bars, {}, start, cutoff)
    assert all(splits.values()) and stats["purged"] >= 4
    for index, rows in enumerate(splits.values()):
        assert all(row["target_end"] < stats["edges"][index + 1] for row in rows)
        state = json.loads(rows[0]["state"])
        assert max(bar["epoch"] + 300 for bar in state["recent_candles"]) <= rows[0]["observation_epoch"]
    changed = bars.copy()
    changed[-1] = replace(changed[-1], close=100)
    altered, _ = dataset.build_splits(changed, {}, start, cutoff)
    assert altered["train"] == splits["train"]
    bars[130] = replace(bars[130], close=bars[129].close)
    _, ties = dataset.build_splits(bars, {}, start, cutoff)
    assert ties["ties"] > 0
    del bars[140]
    _, gaps = dataset.build_splits(bars, {}, start, cutoff)
    assert gaps["gaps"] > 0


def test_manifest_tampering(prepared):
    assert artifacts.verify_dataset(prepared)["counts"]["train"] > 0
    (prepared / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="alterado"):
        artifacts.verify_dataset(prepared)


def test_gpu_check_precedes_any_weight_load(prepared, tmp_path, monkeypatch):
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="CUDA indisponivel"):
        training.cuda_preflight(fake)
    fake.cuda.is_available = lambda: True
    fake.cuda.mem_get_info = lambda: (1, 1)
    with pytest.raises(RuntimeError, match="8 GiB"):
        training.cuda_preflight(fake)
    fake.cuda.mem_get_info = lambda: (16 * 1024**3, 16 * 1024**3)
    fake.cuda.get_device_name = lambda index: "GPU test"
    assert training.cuda_preflight(fake)["device"] == "GPU test"
    monkeypatch.setattr(training, "cuda_preflight", lambda torch: (_ for _ in ()).throw(RuntimeError("no GPU")))
    monkeypatch.setattr(training, "token_preflight", MagicMock(side_effect=AssertionError("nao carregar")))
    with pytest.raises(RuntimeError, match="no GPU"):
        training.train(prepared, tmp_path / "run")


def test_dry_run_uses_full_context_without_loading_weights(prepared, tmp_path, monkeypatch):
    import transformers

    base = tmp_path / "base"
    (base / "encoder").mkdir(parents=True)
    artifacts.write_json(base / "rl_agent_config.json", {"head_max_len": 192})
    artifacts.write_json(base / "encoder/config.json", {"max_position_embeddings": 100000})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path: context_agent().tok)
    result = training.train(prepared, tmp_path / "run", dry_run=True, base=str(base))
    assert result["dry_run"] and result["max_len"] > 512
    assert not (tmp_path / "run").exists()


@pytest.mark.asyncio
async def test_cached_history_validates_ohlc_and_does_not_open_socket(tmp_path):
    bars = synthetic_bars(2)
    path = tmp_path / "candles.parquet"
    rows = [
        {
            "symbol": "R_10",
            "granularity": 300,
            "epoch": b.epoch,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
        }
        for b in bars
    ]
    pl.DataFrame(rows).write_parquet(path)
    assert await dataset.complete_history(path, bars[0].epoch, bars[-1].epoch + 300, {}) == bars
    rows[0]["high"] = 0
    pl.DataFrame(rows).write_parquet(path)
    with pytest.raises(ValueError, match="OHLC"):
        await dataset.complete_history(path, bars[0].epoch, bars[-1].epoch + 300, {})


def test_test_partition_consumed_once_and_promotion_rechecks_integrity(prepared, tmp_path, monkeypatch):
    from src.infrastructure.inference.laya_checkpoint import checkpoint_digest, resolve_approved_checkpoint

    run = tmp_path / "run"
    candidate = run / "seed-42"
    candidate.mkdir(parents=True)
    (candidate / "model.safetensors").write_bytes(b"fake weights")
    identity = checkpoint_digest(candidate)
    artifacts.write_json(
        run / "selection.json",
        {
            "candidate": "seed-42",
            "base_checkpoint": "base",
            "checkpoint_sha256": identity,
            "dataset_sha256": artifacts.file_digest(prepared / "manifest.json"),
        },
    )

    def score(agent, rows):
        return {
            "n": len(rows),
            "accuracy": 0.7 if agent == "base" else 0.8,
            "majority": 0.5,
            "balanced_accuracy": 0.8,
            "recall": {"CALL": 0.8, "PUT": 0.8},
            "brier": 0.15,
        }

    monkeypatch.setattr(evaluation, "score_rows", score)
    report = evaluation.evaluate(prepared, run, loader=lambda path: path)
    assert report["approved"] and report["realized_profit_claim"] is False
    with pytest.raises(FileExistsError):
        evaluation.evaluate(prepared, run, loader=lambda path: path)
    assert evaluation.promote(prepared, run)["approved_checkpoint"] == str(candidate.resolve())
    assert resolve_approved_checkpoint("base", str(candidate))[1] == identity
    (candidate / "model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError):
        evaluation.promote(prepared, run)


def test_scoring_real_probabilities_and_both_classes_required():
    agent = context_agent()

    def predict(text, questions, **kwargs):
        bullish = json.loads(text)["indicators"]["ema_20"] > 1
        return {
            "answers": {
                "direction": {"probabilities": {"CALL": 0.9 if bullish else 0.1, "PUT": 0.1 if bullish else 0.9}}
            },
            "usage": {"truncated": False},
        }

    agent.system_one = predict
    rows = [
        {
            "state": json.dumps({"indicators": {"ema_20": value}}),
            "questions": {
                "direction": {"type": "choice", "instructions": "direction", "criteria": {"CALL": "up", "PUT": "down"}}
            },
            "expected": {"direction": label},
        }
        for value, label in ((2, "CALL"), (0, "PUT"))
    ]
    metrics = evaluation.score_rows(agent, rows)
    assert metrics["accuracy"] == 1 and metrics["brier"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="ambas"):
        evaluation.score_rows(agent, rows[:1])
    agent.system_one = lambda *args, **kwargs: {"usage": {"truncated": False}}
    with pytest.raises(ValueError, match="Probabilidades invalidas"):
        evaluation.score_rows(agent, rows)


def test_three_seeds_use_only_train_and_reserved_calibration(prepared, tmp_path, monkeypatch):
    import laya
    import laya.calibrate
    import laya.train
    import torch

    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    budget = {"max_len": 2048, "head_max_len": 192, "base_checkpoint": str(base), "base_revision": "test"}
    monkeypatch.setattr(training, "token_preflight", lambda *args: budget)
    monkeypatch.setattr(training, "cuda_preflight", lambda *args: {"device": "fake CUDA"})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(laya.train, "load_checkpoint", lambda path: (SimpleNamespace(to=lambda d: None), None, {}))
    monkeypatch.setattr(laya.train, "items_from_rows", lambda tok, rows, *args: (rows, {}))
    seen_seeds = []

    def train_model(model, tok, rows, config, device, max_len, head_max_len, **kwargs):
        assert rows == artifacts.read_rows(prepared / "train.jsonl")
        assert config.epochs == 4 and config.micro_batch == 1 and config.grad_accum == 64
        assert config.amp and config.gradient_checkpointing
        seen_seeds.append(config.seed)
        kwargs["on_epoch_end"](1, 0.5)
        return [0.5]

    monkeypatch.setattr(laya.train, "train_model", train_model)

    def calibrate(model, tok, rows, *args, **kwargs):
        assert rows == artifacts.read_rows(prepared / "calibration.jsonl")
        assert kwargs["batch_size"] == 1
        return ["calibration"]

    monkeypatch.setattr(laya.train, "calibration_records", calibrate)
    monkeypatch.setattr(
        laya.calibrate, "fit_temperature_map", lambda records: {"temperature": [1, 1, 1], "temperature_by_options": {}}
    )

    def save(model, tok, config, path):
        from pathlib import Path

        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_bytes(b"trained")
        artifacts.write_json(target / "rl_agent_config.json", config)

    monkeypatch.setattr(laya.train, "save_checkpoint", save)
    monkeypatch.setattr(laya, "load", lambda path, **kwargs: path)

    def score(agent, rows):
        assert rows == artifacts.read_rows(prepared / "selection.jsonl")
        return {"accuracy": 0.7 if str(agent).endswith("seed-42") else 0.6, "brier": 0.2}

    monkeypatch.setattr(training, "score_rows", score)
    output = tmp_path / "run"
    result = training.train(prepared, output)
    assert seen_seeds == [17, 42, 73]
    assert result["candidate"] == "seed-42"
    assert not (prepared / "test_consumed.json").exists()
    with pytest.raises(ValueError, match="ja existe"):
        training.train(prepared, output)
