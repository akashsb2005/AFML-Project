import csv
import math

import pytest
import torch

from src.training import (
    LOG_FIELDS,
    _set_learning_rate,
    _write_log_header,
    _save_checkpoint,
    evaluate,
    load_config,
    validate_config,
)


def test_project_configs_match_except_attention():
    baseline = load_config(__import__("pathlib").Path("configs/baseline.yaml"))
    efficient = load_config(__import__("pathlib").Path("configs/efficient_attention.yaml"))
    assert baseline["model"]["attention"] == "standard"
    assert efficient["model"]["attention"] == "linear"
    for key in ("vocab_size", "n_layer", "n_head", "d_model", "context_length", "dropout"):
        assert baseline["model"][key] == efficient["model"][key]
    assert baseline["train"] == efficient["train"]
    assert baseline["data"] == efficient["data"]


def test_validate_config_rejects_non_divisible_model_dimensions():
    config = {
        "model": {"vocab_size": 32, "n_layer": 1, "n_head": 3, "d_model": 8,
                  "context_length": 8, "dropout": 0.0, "attention": "standard"},
        "train": {"batch_size": 1, "lr": 1e-3, "weight_decay": 0.0,
                  "warmup_steps": 0, "max_steps": 2, "eval_interval": 1,
                  "eval_batches": 1},
        "data": {},
    }
    with pytest.raises(ValueError, match="divisible"):
        validate_config(config)


def test_warmup_is_linear_then_constant():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.3)
    assert _set_learning_rate(optimizer, 0.3, 0, 3) == pytest.approx(0.1)
    assert _set_learning_rate(optimizer, 0.3, 1, 3) == pytest.approx(0.2)
    assert _set_learning_rate(optimizer, 0.3, 2, 3) == pytest.approx(0.3)
    assert _set_learning_rate(optimizer, 0.3, 8, 3) == pytest.approx(0.3)


class UniformModel(torch.nn.Module):
    vocab_size = 5

    def forward(self, input_ids, attention_mask=None):
        return torch.zeros((*input_ids.shape, self.vocab_size), device=input_ids.device)


def test_evaluate_is_token_weighted_and_restores_training_state():
    model = UniformModel().train()
    batches = [
        {"input_ids": torch.tensor([[1, 2, 0]]),
         "labels": torch.tensor([[2, 3, -100]]),
         "attention_mask": torch.tensor([[1, 1, 0]])},
        {"input_ids": torch.tensor([[1, 2, 3]]),
         "labels": torch.tensor([[2, 3, 4]]),
         "attention_mask": torch.tensor([[1, 1, 1]])},
    ]
    loss, tokens = evaluate(model, batches, torch.device("cpu"))
    assert loss == pytest.approx(math.log(model.vocab_size))
    assert tokens == 5
    assert model.training


def test_training_log_header_contract(tmp_path):
    path = tmp_path / "train.csv"
    _write_log_header(path)
    with path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == LOG_FIELDS


def test_training_log_rejects_incompatible_existing_file(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("step,loss\n1,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Choose a new"):
        _write_log_header(path)



def test_checkpoint_config_loads_through_member3_model_api(tmp_path):
    from src.model import build_model
    from src.model_api import load_checkpoint

    model_config = {
        "vocab_size": 17, "n_layer": 1, "n_head": 1,
        "d_model": 8, "context_length": 8, "dropout": 0.0,
        "attention": "standard",
    }
    config = {
        "model": model_config,
        "train": {"batch_size": 1},
        "data": {"tokenizer": "artifacts/tokenizer/tokenizer.json"},
        "seed": 42,
        "training": {},
    }
    model = build_model(model_config)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    _save_checkpoint(
        path, step=1, tokens_seen=3, model=model, optimizer=optimizer,
        config=config, train_loss=1.0, validation_loss=1.1,
        best_validation_loss=1.1, elapsed_s=2.0,
        data_rng_state=torch.Generator().get_state(),
    )
    loaded, checkpoint = load_checkpoint(str(path), torch.device("cpu"))
    assert loaded(torch.ones((1, 4), dtype=torch.long)).shape == (1, 4, 17)
    assert checkpoint["step"] == 1
    assert checkpoint["tokens_seen"] == 3
