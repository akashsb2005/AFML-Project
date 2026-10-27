"""Contract tests run against the test fixture always, and against Member 1's src/model.py once it exists."""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUILDERS = ["tests.fixtures.tiny_model:build_model"]
if (ROOT / "src" / "model.py").exists() and (ROOT / "src" / "efficient_model.py").exists():
    BUILDERS.append("auto")                      # Member 1's SmallLanguageModel / EfficientSmallLanguageModel

SMALL = dict(vocab_size=97, n_layer=2, n_head=2, d_model=32, context_length=32, dropout=0.0)


@pytest.fixture(params=[(b, a) for b in BUILDERS for a in ("standard", "linear")],
                ids=lambda p: f"{'member1' if p[0] == 'auto' else 'fixture'}-{p[1]}")
def model_and_config(request):
    from src import model_api
    builder, attention = request.param
    cfg = dict(SMALL, attention=attention)
    try:
        model = model_api.build_model(cfg, builder)
    except (ValueError, KeyError, NotImplementedError) as e:
        pytest.skip(f"{builder} does not support attention={attention}: {e}")
    return model.eval(), cfg
