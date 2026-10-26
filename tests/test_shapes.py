import torch


def test_logits_shape(model_and_config):
    model, cfg = model_and_config
    x = torch.randint(0, cfg["vocab_size"], (3, 17))
    assert model(x).shape == (3, 17, cfg["vocab_size"])


def test_full_context_length(model_and_config):
    model, cfg = model_and_config
    x = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]))
    assert model(x).shape == (1, cfg["context_length"], cfg["vocab_size"])
