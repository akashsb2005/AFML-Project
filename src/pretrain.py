"""Pretrain the standard causal softmax-attention model."""
try:
    from .training import run_training
except ImportError:  # Support: python src/pretrain.py
    from training import run_training


if __name__ == "__main__":
    run_training("standard")
