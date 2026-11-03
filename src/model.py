import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import CausalSelfAttention
except ImportError:  # Support direct execution with src/ on sys.path.
    from attention import CausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()
        hidden_dim = 4 * embedding_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.ln1(x), attention_mask=attention_mask)
        return x + self.feed_forward(self.ln2(x))


class SmallLanguageModel(nn.Module):
    """Decoder-only Transformer language model for next-token prediction."""

    def __init__(
        self,
        vocab_size=8000,
        embedding_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=256,
        dropout=0.1,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must match input_ids shape")
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        # Evaluation and generation APIs consume logits when no targets are supplied.
        return (logits, loss) if targets is not None else logits


def build_model(config: dict) -> nn.Module:
    """Build a model from the shared model configuration."""
    attention = config.get("attention", "standard")
    common = {
        "vocab_size": int(config.get("vocab_size", 8000)),
        "embedding_dim": int(config.get("d_model", config.get("embedding_dim", 256))),
        "num_heads": int(config.get("n_head", config.get("num_heads", 4))),
        "num_layers": int(config.get("n_layer", config.get("num_layers", 4))),
        "max_seq_len": int(config.get("context_length", config.get("max_seq_len", 256))),
        "dropout": float(config.get("dropout", 0.1)),
    }
    if attention in {"linear", "causal_linear_elu_plus_one"}:
        try:
            from .efficient_model import EfficientSmallLanguageModel
        except ImportError:  # Support direct execution with src/ on sys.path.
            from efficient_model import EfficientSmallLanguageModel
        return EfficientSmallLanguageModel(**common)
    if attention != "standard":
        raise ValueError(f"Unsupported attention variant: {attention}")
    return SmallLanguageModel(**common)
