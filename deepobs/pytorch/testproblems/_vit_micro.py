"""ViT-Micro: a small Vision Transformer for image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Project image patches to embedding space using a strided convolution.

    Splits an image into non-overlapping square patches and linearly projects
    each patch to ``embed_dim`` dimensions.  The convolution kernel size and
    stride are both equal to ``patch_size``, so there is no overlap between
    patches.

    Args:
        image_size (int): Input image side length. Must be divisible by
            ``patch_size``.
        patch_size (int): Side length of each square patch.
        in_channels (int): Number of input image channels (e.g. 3 for RGB).
        embed_dim (int): Output embedding dimension per patch.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image to patch token sequence.

        Args:
            x (torch.Tensor): Input images of shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Patch embeddings of shape
                ``(B, num_patches, embed_dim)``.
        """
        x = self.proj(x)          # (B, embed_dim, H//p, W//p)
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (full, non-causal).

    Args:
        embed_dim (int): Token embedding dimension.
        num_heads (int): Number of attention heads.
            ``embed_dim`` must be divisible by ``num_heads``.
        dropout (float): Dropout probability on attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Token sequence of shape ``(B, N, embed_dim)``.

        Returns:
            torch.Tensor: Output of shape ``(B, N, embed_dim)``.
        """
        B, N, C = x.shape

        # Compute Q, K, V and split heads
        qkv = self.qkv(x)                                  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)                            # each (B, H, N, d)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale      # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate values and merge heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)    # (B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LayerNorm ViT encoder block.

    Applies multi-head self-attention followed by a 2-layer MLP with GELU
    activation.  Both sub-layers use a pre-normalisation residual connection.

    Args:
        embed_dim (int): Token embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dimension to ``embed_dim``.
            Defaults to 4.0.
        dropout (float): Dropout probability applied inside the MLP and on
            attention weights.  Defaults to 0.0.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-LN residual connections.

        Args:
            x (torch.Tensor): Token sequence of shape ``(B, N, embed_dim)``.

        Returns:
            torch.Tensor: Output of shape ``(B, N, embed_dim)``.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTMicro(nn.Module):
    """ViT-Micro: a ~2.84M-parameter Vision Transformer for image classification.

    Architecture overview:

    1. **Patch embedding** – a strided convolution splits the image into
       non-overlapping patches and projects each to ``embed_dim`` dimensions.
    2. **[CLS] token** – a learnable token prepended to the patch sequence.
    3. **Positional embeddings** – learnable 1-D embeddings added to every
       token (including [CLS]).
    4. **Transformer encoder** – ``depth`` pre-LN blocks, each with multi-head
       self-attention and a 2-layer GELU MLP.
    5. **Classification head** – a linear layer applied to the final [CLS]
       representation.

    Default configuration (ViT-Micro-192):

    ====================== =============================
    Hyper-parameter         Value
    ====================== =============================
    ``image_size``          96
    ``patch_size``          16
    ``embed_dim``           192
    ``depth``               6
    ``num_heads``           3
    ``mlp_ratio``           4.0
    ``dropout``             0.1
    num_patches             36  (= (96/16)²)
    **~Parameters**         **2,844,005**
    ====================== =============================

    Args:
        image_size (int): Input image side length. Must be divisible by
            ``patch_size``. Defaults to 96.
        patch_size (int): Side length of each square patch. Defaults to 16.
        in_channels (int): Number of input image channels. Defaults to 3.
        num_classes (int): Number of output classes. Defaults to 101.
        embed_dim (int): Token embedding dimension. Defaults to 192.
        depth (int): Number of transformer encoder blocks. Defaults to 6.
        num_heads (int): Number of attention heads per block. Defaults to 3.
        mlp_ratio (float): Ratio of MLP hidden dimension to ``embed_dim``.
            Defaults to 4.0.
        dropout (float): Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 101,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.emb_drop = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialise weights following the original ViT paper."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input images of shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Class logits of shape ``(B, num_classes)``.
        """
        B = x.size(0)

        # 1. Patch embedding
        x = self.patch_embed(x)                             # (B, N, D)

        # 2. Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                     # (B, N+1, D)

        # 3. Add positional embeddings
        x = x + self.pos_embed
        x = self.emb_drop(x)

        # 4. Transformer encoder
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 5. Classification from [CLS] token
        cls_out = x[:, 0]                                   # (B, D)
        return self.head(cls_out)                           # (B, num_classes)
