"""GPT-Micro: a small decoder-only transformer for character-level language modeling."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a triangular mask.

    Args:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads.
        max_seq_len (int): Maximum sequence length (used to build the causal mask).
        dropout (float): Dropout probability on attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Fused QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        # Causal mask (lower-triangular); registered as a buffer so it moves with .to(device)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Shape (B, T, d_model).

        Returns:
            torch.Tensor: Shape (B, T, d_model).
        """
        B, T, C = x.size()

        # Project and split into Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)  # each (B, T, C)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block with causal self-attention and GELU FFN.

    Args:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads.
        d_ff (int): Feed-forward hidden dimension.
        max_seq_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-LN residual connections.

        Args:
            x (torch.Tensor): Shape (B, T, d_model).

        Returns:
            torch.Tensor: Shape (B, T, d_model).
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTMicro(nn.Module):
    """GPT-Micro: a small (~1.24M parameter) decoder-only transformer.

    Architecture:
    - Token embedding: vocab_size × d_model
    - Positional embedding: max_seq_len × d_model
    - N transformer blocks (pre-LN, causal attention, GELU FFN)
    - Final layer norm
    - LM head tied to the token embedding weights

    Default hyperparameters:
        n_layers=6, d_model=128, n_heads=4, d_ff=512,
        max_seq_len=128, vocab_size=256, dropout=0.1

    Args:
        vocab_size (int): Vocabulary size. Defaults to 256 (byte-level).
        n_layers (int): Number of transformer blocks. Defaults to 6.
        d_model (int): Model dimension. Defaults to 128.
        n_heads (int): Number of attention heads. Defaults to 4.
        d_ff (int): Feed-forward hidden dimension. Defaults to 512.
        max_seq_len (int): Maximum sequence length. Defaults to 128.
        dropout (float): Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        n_layers: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)

        # LM head — weight-tied to token embedding
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Token ids of shape (B, T).

        Returns:
            torch.Tensor: Logits of shape (B, T, vocab_size).
        """
        B, T = x.size()
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        )

        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)

        tok = self.token_emb(x)  # (B, T, d_model)
        pos = self.pos_emb(positions)  # (1, T, d_model)
        h = self.emb_drop(tok + pos)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.lm_head(h)  # (B, T, vocab_size)

        return logits
