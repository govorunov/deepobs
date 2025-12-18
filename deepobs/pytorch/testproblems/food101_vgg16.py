"""VGG 16 architecture for Food-101."""

import torch
import torch.nn.functional as F

from ._vgg import vgg16
from ..datasets.food101 import Food101
from .testproblem import TestProblem


class food101_vgg16(TestProblem):
    """DeepOBS test problem class for the VGG 16 network on Food-101.

    Food-101 consists of 101 food categories with 750 training and 250 test
    images per category. Images are resized and cropped to 224x224 to match
    the original VGG network architecture designed for ImageNet.

    Details about the architecture can be found in the original paper:
    https://arxiv.org/abs/1409.1556

    VGG 16 consists of 16 weight layers, of mostly convolutions. This version
    uses Batch Normalization after each convolutional layer (momentum=0.1) and
    reduced dropout (0.2) in the fully connected layers.

    The model uses cross-entropy loss. A weight decay is used on the weights
    (but not the biases) which defaults to 5e-4.

    The reference training parameters are batch size = 128, num_epochs = 50-100
    using SGD with momentum.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
            is used on the weights but not the biases. Defaults to 5e-4.
        device (str or torch.device, optional): Device to use.
    """

    def __init__(self, batch_size, weight_decay=5e-4, device=None):
        """Create a new VGG 16 test problem instance on Food-101.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float): Weight decay factor. Defaults to 5e-4.
            device (str or torch.device, optional): Device to use.
        """
        super(food101_vgg16, self).__init__(batch_size, weight_decay, device)

    def set_up(self):
        """Set up the VGG 16 test problem on Food-101."""
        # Initialize dataset
        self.dataset = Food101(self._batch_size)

        # Initialize model with BatchNorm and reduced dropout
        self.model = vgg16(num_outputs=101, input_size=(224, 224),
                          batch_norm=True, dropout=0.2, bn_momentum=0.1)
        self.model.to(self.device)

    def _compute_loss(self, outputs, targets, reduction='mean'):
        """Compute cross-entropy loss.

        Args:
            outputs (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Ground truth labels (one-hot or class indices).
            reduction (str): 'mean' or 'none'.

        Returns:
            torch.Tensor: Loss value(s).
        """
        # Handle one-hot encoded targets
        if targets.dim() == 2 and targets.size(1) > 1:
            targets = targets.argmax(dim=1)

        return F.cross_entropy(outputs, targets, reduction=reduction)
