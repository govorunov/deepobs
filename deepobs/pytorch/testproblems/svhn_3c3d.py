"""A vanilla CNN architecture for SVHN."""

import torch
import torch.nn.functional as F

from ._3c3d import CNN3c3d
from ..datasets.svhn import svhn
from .testproblem import TestProblem


class svhn_3c3d(TestProblem):
    """DeepOBS test problem class for a three convolutional and three dense
    layered neural network on SVHN.

    The network consists of:
    - three conv layers with ReLUs, each followed by max-pooling
    - two fully-connected layers with 512 and 256 units and ReLU activation
    - 10-unit output layer with softmax
    - cross-entropy loss
    - L2 regularization on the weights (but not the biases) with a default
      factor of 0.002

    The weight matrices are initialized using Xavier initialization and the biases
    are initialized to 0.0.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
            is used on the weights but not the biases. Defaults to 0.002.
        device (str or torch.device, optional): Device to use.
    """

    def __init__(self, batch_size, weight_decay=0.002, device=None):
        """Create a new 3c3d test problem instance on SVHN.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float): Weight decay factor. Defaults to 0.002.
            device (str or torch.device, optional): Device to use.
        """
        super(svhn_3c3d, self).__init__(batch_size, weight_decay, device)

    def set_up(self):
        """Set up the vanilla CNN test problem on SVHN."""
        # Initialize dataset
        self.dataset = svhn(self._batch_size)

        # Initialize model
        self.model = CNN3c3d(num_outputs=10)
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
