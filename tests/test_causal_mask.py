import torch


def test_future_tokens_do_not_change_past_logits(model_and_config):
    """logits[:, :t+1] must be identical when only tokens after position t are changed."""
    model, cfg = model_and_config
    torch.manual_seed(0)
    T, t = 20, 9
    a = torch.randint(0, cfg["vocab_size"], (2, T))
    b = a.clone()
    b[:, t + 1:] = torch.randint(0, cfg["vocab_size"], (2, T - t - 1))
    with torch.no_grad():
        la, lb = model(a), model(b)
    assert torch.allclose(la[:, :t + 1], lb[:, :t + 1], atol=1e-5)
    assert not torch.allclose(la[:, t + 1:], lb[:, t + 1:], atol=1e-5)


def test_prefix_consistency(model_and_config):
    """Running on a prefix gives the same logits as the prefix positions of the full run."""
    model, cfg = model_and_config
    x = torch.randint(0, cfg["vocab_size"], (1, 24))
    with torch.no_grad():
        assert torch.allclose(model(x[:, :11]), model(x)[:, :11], atol=1e-5)
