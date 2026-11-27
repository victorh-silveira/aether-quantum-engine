"""Testes do modo rapido Laya: presets, lote, warm start, progresso e retomada."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.laya_workflow import (
    __main__ as cli_main,
    artifacts,
    batch_tuning,
    fast_backend,
    run_guard,
    training_run,
    training_state,
)


def _make_checkpoint(directory: Path, *, max_len: int = 2048, encoder_type: str = "modernbert", tok_text: str = "{}"):
    """Cria estrutura minima de checkpoint Laya para testes de compatibilidade."""
    (directory / "encoder").mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer").mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").write_bytes(b"weights")
    artifacts.write_json(
        directory / "rl_agent_config.json",
        {
            "encoder": "answerdotai/ModernBERT-large",
            "head_layers": 2,
            "head_max_len": 192,
            "max_prefixes": 6,
            "max_len": max_len,
        },
    )
    artifacts.write_json(
        directory / "encoder/config.json",
        {"model_type": encoder_type, "hidden_size": 1024, "num_hidden_layers": 28, "vocab_size": 50368},
    )
    (directory / "tokenizer/tokenizer.json").write_text(tok_text, encoding="utf-8")
    return directory


class _TinyHeadModel(torch.nn.Module):
    """Modelo compacto com encoder congelavel e cabeca treinavel para testar retomada."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 2)
        self.head_checkpointing = False

    def forward(self, x):
        return self.head(self.encoder(x))


def test_presets_and_remaining_epochs():
    fast_fresh = training_run.schedule("fast", 0)
    assert fast_fresh == {
        "preset": "fast",
        "epochs": 2,
        "seeds": [17],
        "inherited_epochs": 0,
        "remaining_epochs": 2,
    }
    fast_warm = training_run.schedule("fast", 1)
    assert fast_warm["remaining_epochs"] == 1 and fast_warm["inherited_epochs"] == 1
    complete = training_run.schedule("complete", 0)
    assert complete == {
        "preset": "complete",
        "epochs": 4,
        "seeds": [17, 42, 73],
        "inherited_epochs": 0,
        "remaining_epochs": 4,
    }
    with pytest.raises(ValueError, match="Preset invalido"):
        training_run.schedule("turbo", 0)
    with pytest.raises(ValueError, match="epocas herdadas"):
        training_run.schedule("fast", 2)


def test_choose_batch_selects_largest_and_backs_off_on_oom():
    def probe_all_ok(size):
        return {"free_margin_bytes": 2 * 1024**3, "examples_per_second": float(size)}

    selected, records = batch_tuning.choose_batch(probe_all_ok)
    assert selected == 4 and [r["batch"] for r in records] == [1, 2, 4]

    def probe_oom_at_4(size):
        if size == 4:
            raise torch.OutOfMemoryError("OOM")
        return {"free_margin_bytes": 2 * 1024**3, "examples_per_second": float(size)}

    selected, records = batch_tuning.choose_batch(probe_oom_at_4)
    assert selected == 2 and 64 // selected == 32
    assert records[-1] == {"batch": 4, "oom": True}

    def probe_low_margin_at_2(size):
        margin = 2 * 1024**3 if size == 1 else 512 * 1024**2
        return {"free_margin_bytes": margin, "examples_per_second": float(size)}

    selected, _ = batch_tuning.choose_batch(probe_low_margin_at_2)
    assert selected == 1 and 64 // selected == 64

    with pytest.raises(RuntimeError, match="1 GiB"):
        batch_tuning.choose_batch(lambda _size: {"free_margin_bytes": 128, "examples_per_second": 1.0})


def test_tune_batch_restores_weights_and_rng(monkeypatch):
    model = _TinyHeadModel()
    for param in model.encoder.parameters():
        param.requires_grad = False
    initial_head = model.head.weight.detach().clone()
    initial_bias = model.head.bias.detach().clone()
    encoded = [{"ids": [1, 2, 3]} for _ in range(4)]
    config = SimpleNamespace(head_lr=0.01)

    monkeypatch.setattr(
        fast_backend, "batch_loss", lambda m, chunk, cfg, dev, epoch: m(torch.ones(len(chunk), 4)).sum()
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (4 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 1024**2)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2 * 1024**2)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.amp,
        "GradScaler",
        lambda *_a, **_k: SimpleNamespace(
            scale=lambda loss: loss,
            step=lambda opt: opt.step(),
            update=lambda: None,
        ),
    )
    torch.manual_seed(123)
    expected_rand = torch.rand(3)
    torch.manual_seed(123)

    selected, records = batch_tuning.tune_batch(model, encoded, config, torch.device("cpu"))
    assert selected == 4 and len(records) == 3
    assert torch.equal(model.head.weight.detach(), initial_head)
    assert torch.equal(model.head.bias.detach(), initial_bias)
    assert torch.equal(torch.rand(3), expected_rand)


def test_validate_warm_start_accepts_compatible_and_rejects_broken(tmp_path):
    base = _make_checkpoint(tmp_path / "base", max_len=2048)
    legacy = _make_checkpoint(tmp_path / "legacy", max_len=2048)
    budget = {"max_len": 1500}
    digest = training_run.validate_warm_start(legacy, base, budget)
    assert len(digest) == 64

    with pytest.raises(ValueError, match="diretorio ausente"):
        training_run.validate_warm_start(tmp_path / "missing", base, budget)
    with pytest.raises(ValueError, match="pesos treinados"):
        training_run.validate_warm_start(base, base, budget)

    bad_tok = _make_checkpoint(tmp_path / "bad_tok", tok_text='{"diff": true}')
    with pytest.raises(ValueError, match="Tokenizer legado incompativel"):
        training_run.validate_warm_start(bad_tok, base, budget)

    bad_ctx = _make_checkpoint(tmp_path / "bad_ctx", max_len=512)
    with pytest.raises(ValueError, match="Contexto legado menor"):
        training_run.validate_warm_start(bad_ctx, base, budget)

    bad_enc = _make_checkpoint(tmp_path / "bad_enc", encoder_type="bert")
    with pytest.raises(ValueError, match="Encoder legado incompativel"):
        training_run.validate_warm_start(bad_enc, base, budget)

    empty_weights = _make_checkpoint(tmp_path / "empty_weights")
    (empty_weights / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="Checkpoint legado incompleto"):
        training_run.validate_warm_start(empty_weights, base, budget)


def test_cli_warm_start_auto_routes_absent_valid_and_broken(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli_main, "train", lambda *args, **kwargs: calls.append(kwargs) or {"ok": True})

    absent = tmp_path / "absent"
    monkeypatch.setattr(
        sys,
        "argv",
        ["laya_workflow", "train", "--data", str(tmp_path), "--preset", "fast", "--warm-start-auto", str(absent)],
    )
    cli_main.main()
    assert calls[-1]["warm_start"] is None and calls[-1]["completed_epochs"] == 0

    broken = tmp_path / "broken"
    broken.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["laya_workflow", "train", "--data", str(tmp_path), "--preset", "fast", "--warm-start-auto", str(broken)],
    )
    cli_main.main()
    assert calls[-1]["warm_start"] == broken and calls[-1]["completed_epochs"] == 1


def test_training_progress_controlled_clock():
    now = [0.0]
    messages = []
    progress = fast_backend.TrainingProgress(128, clock=lambda: now[0], emit=lambda msg, **_kw: messages.append(msg))

    now[0] = 6.0
    progress.update(32, 1, 2, 32, 128)
    assert "eta=calculando" in messages[-1] and "samples=32/128" in messages[-1]

    now[0] = 8.0
    progress.update(16, 1, 2, 48, 128)
    assert len(messages) == 1

    now[0] = 16.0
    progress.update(16, 1, 2, 64, 128)
    assert len(messages) == 2
    assert "speed=4.00/s" in messages[-1] and "elapsed=0.27min" in messages[-1] and "eta=0.27min" in messages[-1]

    now[0] = 17.0
    progress.update(64, 1, 2, 128, 128)
    assert len(messages) == 3 and "samples=128/128" in messages[-1]


def test_checkpoint_resume_preserves_state_and_optimizer(tmp_path, monkeypatch):
    def fake_save_ckpt(_model, _tok, cfg, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        artifacts.write_json(Path(path) / "rl_agent_config.json", cfg)

    monkeypatch.setattr("laya.train.save_checkpoint", fake_save_ckpt)
    monkeypatch.setattr(
        fast_backend, "batch_loss", lambda m, chunk, _cfg, _dev, epoch: (m(torch.stack(chunk)) ** 2).sum()
    )

    model_a = _TinyHeadModel()
    model_b = _TinyHeadModel()
    model_b.load_state_dict(model_a.state_dict())
    encoded = [torch.full((4,), float(i + 1)) for i in range(128)]
    cfg = {"max_len": 64, "head_max_len": 16}
    options = SimpleNamespace(
        epochs=2,
        micro_batch=2,
        grad_accum=32,
        encoder_lr=1e-4,
        head_lr=1e-2,
        min_lr=1e-6,
        weight_decay=0.0,
        grad_clip=1.0,
        amp=False,
        gradient_checkpointing=False,
        freeze_encoder=True,
        seed=17,
    )

    calls = [0]

    def pause_after_first_window():
        calls[0] += 1
        return [calls[0] >= 1]

    from contextlib import nullcontext

    target = tmp_path / "seed-17"
    monkeypatch.setattr(fast_backend, "controlled_stop", lambda: nullcontext([True]))
    with pytest.raises(KeyboardInterrupt):
        fast_backend.run_training(
            model_a, None, encoded, options, torch.device("cpu"), target=target, cfg=cfg, start_epoch=1
        )

    _, loaded_state = training_state.load_training_checkpoint(target)
    assert loaded_state["epoch"] == 1 and loaded_state["position"] == 64

    monkeypatch.setattr(fast_backend, "controlled_stop", lambda: nullcontext([False]))
    history_resumed = fast_backend.run_training(
        model_a, None, encoded, options, torch.device("cpu"), target=target, cfg=cfg, start_epoch=1, state=loaded_state
    )
    history_uninterrupted = fast_backend.run_training(
        model_b, None, encoded, options, torch.device("cpu"), target=tmp_path / "full", cfg=cfg, start_epoch=1
    )
    assert history_resumed == pytest.approx(history_uninterrupted, rel=1e-6)
    assert torch.allclose(model_a.head.weight, model_b.head.weight, atol=1e-6)


def test_run_guard_rejects_active_other_training(tmp_path, monkeypatch):
    monkeypatch.setattr(run_guard, "other_training_pids", lambda: [99999])
    with pytest.raises(RuntimeError, match="Outro treinamento ativo"), run_guard.training_lock(tmp_path):
        pass


def test_collect_calibration_records_uses_amp_and_emits_progress(monkeypatch):
    seen_amp = []
    messages = []
    now = [0.0]

    monkeypatch.setattr("laya.train.encode_item", lambda _tok, item, *_a: item)
    monkeypatch.setattr("laya.train.collate_items", lambda batch, _pad: batch[0])

    def fake_forward(_model, batch, _device, *, amp: bool, detach_encoder: bool):
        assert detach_encoder is False
        seen_amp.append(amp)
        return torch.tensor([[0.8, 0.2, 0.0] for _ in batch], dtype=torch.float32)

    monkeypatch.setattr("laya.train._forward", fake_forward)

    model = _TinyHeadModel()
    tok = SimpleNamespace(pad_token_id=0)
    items = [
        {"qtype": 0, "k": 2, "target": [1.0, 0.0]},
        {"qtype": 0, "k": 2, "target": [0.0, 1.0]},
        {"qtype": 0, "k": 2, "target": [1.0, 0.0]},
    ]

    def tick():
        now[0] += 6.0
        return now[0]

    records_cuda = fast_backend.collect_calibration_records(
        model,
        tok,
        items,
        torch.device("cuda"),
        2048,
        192,
        batch_size=2,
        amp=True,
        clock=tick,
        emit=lambda msg, **_kw: messages.append(msg),
    )
    assert seen_amp == [True, True]
    assert len(records_cuda) == 3
    assert records_cuda[0][0] == 0 and records_cuda[0][3] == 2
    assert len(records_cuda[0][1]) == 2
    assert "CALIB_PROGRESS | samples=2/3" in messages[0] and "eta=" in messages[0]
    assert "CALIB_PROGRESS | samples=3/3" in messages[-1]

    seen_amp.clear()
    fast_backend.collect_calibration_records(
        model,
        tok,
        items[:1],
        torch.device("cpu"),
        2048,
        192,
        batch_size=2,
        amp=True,
        clock=lambda: 10.0,
        emit=lambda _msg, **_kw: None,
    )
    assert seen_amp == [False]


def test_promote_operator_override_preserves_integrity_and_unapproved_flag(tmp_path):
    from src.infrastructure.inference.laya_checkpoint import checkpoint_digest, resolve_approved_checkpoint
    from tools.laya_workflow import evaluation

    data = tmp_path / "data"
    run = tmp_path / "run"
    candidate = run / "seed-17"
    data.mkdir()
    candidate.mkdir(parents=True)
    for name in (
        "train.jsonl",
        "calibration.jsonl",
        "selection.jsonl",
        "test.jsonl",
        "questions.json",
        "settings.json",
    ):
        (data / name).write_text("{}\n", encoding="utf-8")
    (data / "candles.parquet").write_bytes(b"parquet")
    artifacts.write_json(
        data / "manifest.json",
        {
            "files": {
                name: artifacts.file_digest(data / name)
                for name in (
                    "candles.parquet",
                    "questions.json",
                    "settings.json",
                    "train.jsonl",
                    "calibration.jsonl",
                    "selection.jsonl",
                    "test.jsonl",
                )
            }
        },
    )
    (candidate / "model.safetensors").write_bytes(b"weights")
    digest = checkpoint_digest(candidate)
    dataset_sha = artifacts.file_digest(data / "manifest.json")
    artifacts.write_json(
        run / "selection.json",
        {"candidate": "seed-17", "checkpoint_sha256": digest, "dataset_sha256": dataset_sha},
    )
    artifacts.write_json(
        run / "evaluation.json",
        {
            "baseline": {
                "n": 100,
                "accuracy": 0.50,
                "majority": 0.50,
                "balanced_accuracy": 0.50,
                "recall": {"CALL": 0.5, "PUT": 0.5},
                "brier": 0.25,
            },
            "candidate": {
                "n": 100,
                "accuracy": 0.49,
                "majority": 0.50,
                "balanced_accuracy": 0.49,
                "recall": {"CALL": 0.44, "PUT": 0.54},
                "brier": 0.26,
            },
            "checkpoint_sha256": digest,
            "dataset_sha256": dataset_sha,
        },
    )
    with pytest.raises(ValueError, match="nao aprovado"):
        evaluation.promote(data, run)
    promoted = evaluation.promote(data, run, allow_unapproved_override=True)
    assert promoted["approved"] is False
    assert promoted["operator_override"] is True
    assert resolve_approved_checkpoint("base", str(candidate)) == (str(candidate), digest)
    (candidate / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="alterados"):
        evaluation.promote(data, run, allow_unapproved_override=True)
