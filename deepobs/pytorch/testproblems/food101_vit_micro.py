"""ViT-Micro architecture for Food-101."""

import torch.nn.functional as F

from ._vit_micro import ViTMicro
from ..datasets.food101 import Food101
from .testproblem import TestProblem


class food101_vit_micro(TestProblem):
    """DeepOBS test problem class for ViT-Micro on Food-101.

    Food-101 consists of 101 food categories with 750 training and 250 test
    images per class (75,750 train / 25,250 test total).  Images are resized
    and cropped to 96×96 to suit the micro-scale ViT architecture.

    Architecture (ViT-Micro-192):

    - Patch size 16×16 → 36 non-overlapping patches per image
    - Embed dimension 192, 6 transformer encoder blocks, 3 attention heads
    - MLP expansion ratio 4 (hidden dim 768)
    - Learnable [CLS] token and positional embeddings
    - Classification head applied to the final [CLS] representation
    - Dropout 0.1 throughout
    - **~2.84 M parameters**

    The model is the first transformer-based (ViT) test problem in DeepOBS,
    providing a benchmark that exercises attention-based optimisation dynamics
    absent from all-convolutional and dense baselines.

    Loss function: cross-entropy.

    The default weight decay is ``1e-4`` (applied via the optimizer's
    ``weight_decay`` argument).  The model does **not** add manual L2 loss on
    top of that.

    Reference training parameters:

    - Batch size: 128
    - Epochs: 50–100
    - AdamW with ``learning_rate ≈ 3e-4`` and ``weight_decay = 1e-4`` works
      well as a baseline.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float): Weight decay factor passed to the optimizer.
            Defaults to ``1e-4``.
        device (str or torch.device, optional): Device to use. Defaults to
            ``'mps'`` on Apple Silicon, ``'cuda'`` if available, else
            ``'cpu'``.

    Attributes:
        dataset: The :class:`~deepobs.pytorch.datasets.food101.Food101`
            instance configured for 96×96 images.
        model: The :class:`~deepobs.pytorch.testproblems._vit_micro.ViTMicro`
            model (``nn.Module``).
        device: The device where computations are performed.
    """

    def __init__(self, batch_size, weight_decay=1e-4, device=None):
        """Create a new ViT-Micro test problem instance on Food-101.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float): Weight decay factor. Defaults to ``1e-4``.
            device (str or torch.device, optional): Device to use.
        """
        super(food101_vit_micro, self).__init__(batch_size, weight_decay, device)

    def set_up(self):
        """Set up the ViT-Micro test problem on Food-101."""
        # 96×96 images: proportionally scaled Food-101 transforms
        self.dataset = Food101(self._batch_size, target_size=96)

        # ViT-Micro with 101 output classes
        self.model = ViTMicro(
            image_size=96,
            patch_size=16,
            in_channels=3,
            num_classes=101,
            embed_dim=192,
            depth=6,
            num_heads=3,
            mlp_ratio=4.0,
            dropout=0.1,
        )
        self.model.to(self.device)

    def _compute_loss(self, outputs, targets, reduction='mean'):
        """Compute cross-entropy loss.

        Args:
            outputs (torch.Tensor): Model logits of shape
                ``(batch_size, num_classes)``.
            targets (torch.Tensor): Ground truth class indices of shape
                ``(batch_size,)``.
            reduction (str): ``'mean'`` or ``'none'``. Defaults to
                ``'mean'``.

        Returns:
            torch.Tensor: Scalar loss (``reduction='mean'``) or per-example
                losses of shape ``(batch_size,)`` (``reduction='none'``).
        """
        # Handle one-hot encoded targets
        if targets.dim() == 2 and targets.size(1) > 1:
            targets = targets.argmax(dim=1)

        return F.cross_entropy(outputs, targets, reduction=reduction)
