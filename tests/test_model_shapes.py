import sys
from pathlib import Path
import torch

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model import SmallLanguageModel
from efficient_model import EfficientSmallLanguageModel

def test_standard_model_shape():
    model = SmallLanguageModel(vocab_size=128, embedding_dim=32, num_heads=4, num_layers=2, max_seq_len=16)
    x = torch.randint(0, 128, (2, 8))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 8, 128)
    assert loss is not None and torch.isfinite(loss)

def test_efficient_model_shape():
    model = EfficientSmallLanguageModel(vocab_size=128, embedding_dim=32, num_heads=4, num_layers=2, max_seq_len=16)
    x = torch.randint(0, 128, (2, 8))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 8, 128)
    assert loss is not None and torch.isfinite(loss)
