import math

import torch
import torch.nn.functional as F


def _loss(model, x):
    logits = model(x[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))  # shifted by one


def test_initial_loss_is_finite_scalar_near_log_vocab(model_and_config):
    model, cfg = model_and_config
    torch.manual_seed(0)
    loss = _loss(model, torch.randint(0, cfg["vocab_size"], (4, 17)))
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert abs(loss.item() - math.log(cfg["vocab_size"])) < 1.5   # random init ~ uniform prediction


def test_gradients_reach_every_parameter(model_and_config):
    model, cfg = model_and_config
    model.train()
    _loss(model, torch.randint(0, cfg["vocab_size"], (4, 17))).backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, f"no/invalid gradient for {missing}"


def test_can_overfit_tiny_batch(model_and_config):
    model, cfg = model_and_config
    torch.manual_seed(0)
    model.train()
    x = torch.randint(0, cfg["vocab_size"], (2, 17))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = _loss(model, x).item()
    for _ in range(150):
        loss = _loss(model, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.25 * first, f"loss {first:.3f} -> {loss.item():.3f}"
