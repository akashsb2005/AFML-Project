"""Pretrain the causal linear-attention model."""
try:
    from .training import run_training
except ImportError:  # Support: python src/pretrain_efficient.py
    from training import run_training


if __name__ == "__main__":
    run_training("linear")
