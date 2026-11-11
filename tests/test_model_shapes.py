import sys
from pathlib import Path

import torch

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model import SmallLanguageModel
from efficient_model import EfficientSmallLanguageModel


def test_standard_model_shape():
    model = SmallLanguageModel(
        vocab_size=128, embedding_dim=32, num_heads=4,
        num_layers=2, max_seq_len=16,
    )
    x = torch.randint(0, 128, (2, 8))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 8, 128)
    assert loss is not None and torch.isfinite(loss)
    assert model(x).shape == (2, 8, 128)


def test_efficient_model_shape():
    model = EfficientSmallLanguageModel(
        vocab_size=128, embedding_dim=32, num_heads=4,
        num_layers=2, max_seq_len=16,
    )
    x = torch.randint(0, 128, (2, 8))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 8, 128)
    assert loss is not None and torch.isfinite(loss)
    assert model(x).shape == (2, 8, 128)


def test_standard_model_accepts_padding_mask_and_ignores_padding_labels():
    model = SmallLanguageModel(
        vocab_size=32, embedding_dim=16, num_heads=4,
        num_layers=1, max_seq_len=8, dropout=0.0,
    )
    input_ids = torch.tensor([[2, 5, 7, 0], [2, 4, 0, 0]])
    labels = torch.tensor([[5, 7, 3, -100], [4, 3, -100, -100]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    logits, loss = model(input_ids, targets=labels, attention_mask=mask)
    assert logits.shape == (2, 4, 32)
    assert torch.isfinite(loss)


def test_standard_model_rejects_mismatched_attention_mask():
    model = SmallLanguageModel(
        vocab_size=32, embedding_dim=16, num_heads=4,
        num_layers=1, max_seq_len=8,
    )
    input_ids = torch.ones((2, 4), dtype=torch.long)
    bad_mask = torch.ones((2, 3), dtype=torch.long)
    try:
        model(input_ids, attention_mask=bad_mask)
    except ValueError as exc:
        assert "attention_mask" in str(exc)
    else:
        raise AssertionError("Expected mismatched attention mask to fail")


def test_model_factory_builds_both_attention_variants():
    from model import build_model

    common = {
        "vocab_size": 64,
        "d_model": 32,
        "n_head": 4,
        "n_layer": 1,
        "context_length": 12,
        "dropout": 0.0,
    }
    standard = build_model({**common, "attention": "standard"})
    linear = build_model({**common, "attention": "linear"})

    assert isinstance(standard, SmallLanguageModel)
    assert isinstance(linear, EfficientSmallLanguageModel)
    assert standard(torch.ones((1, 5), dtype=torch.long)).shape == (1, 5, 64)
    assert linear(torch.ones((1, 5), dtype=torch.long)).shape == (1, 5, 64)



def test_efficient_model_rejects_all_padding_attention_mask():
    model = EfficientSmallLanguageModel(
        vocab_size=32, embedding_dim=16, num_heads=4,
        num_layers=1, max_seq_len=8,
    )
    input_ids = torch.ones((2, 4), dtype=torch.long)
    bad_mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])
    try:
        model(input_ids, attention_mask=bad_mask)
    except ValueError as exc:
        assert "at least one valid token" in str(exc)
    else:
        raise AssertionError("Expected an all-padding sequence to fail")


def test_efficient_model_rejects_mismatched_target_shape():
    model = EfficientSmallLanguageModel(
        vocab_size=32, embedding_dim=16, num_heads=4,
        num_layers=1, max_seq_len=8,
    )
    input_ids = torch.ones((2, 4), dtype=torch.long)
    targets = torch.ones((2, 3), dtype=torch.long)
    try:
        model(input_ids, targets=targets)
    except ValueError as exc:
        assert "targets" in str(exc)
    else:
        raise AssertionError("Expected mismatched target shape to fail")
