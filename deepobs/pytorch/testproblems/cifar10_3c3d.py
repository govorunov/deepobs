"""3 Conv + 3 Dense test problem on CIFAR-10 for PyTorch DeepOBS."""

import torch
import torch.nn.functional as F

from .testproblem import TestProblem
from ._3c3d import ThreeC3D
from ..datasets.cifar10 import CIFAR10


class cifar10_3c3d(TestProblem):
    """DeepOBS test problem class for a three convolutional and three dense network on CIFAR-10.

    The network consists of:
        - Three conv layers with ReLUs, each followed by max-pooling
        - Two fully-connected layers with 512 and 256 units and ReLU activation
        - 10-unit output layer with softmax (via cross-entropy loss)
        - Cross-entropy loss
        - L2 regularization on the weights (but not the biases) with a default
          factor of 0.002

    The weight matrices are initialized using Xavier initialization and the biases
    are initialized to 0.0.

    A working training setting is batch size = 128, num_epochs = 100 and
    SGD with learning rate of 0.01.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float, optional): Weight decay factor. Weight decay (L2-regularization)
            is used on the weights but not the biases. Defaults to 0.002.
        device (str or torch.device, optional): Device to use. Defaults to
            'cuda' if available, else 'cpu'.

    Attributes:
        dataset: The CIFAR-10 dataset instance.
        model: The 3C3D model (nn.Module).
        device: The device where computations are performed.
    """

    def __init__(self, batch_size, weight_decay=0.002, device=None):
        """Create a new 3C3D test problem on CIFAR-10.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float, optional): Weight decay factor. Defaults to 0.002.
            device (str or torch.device, optional): Device to use.
        """
        super(cifar10_3c3d, self).__init__(batch_size, weight_decay, device)

    def set_up(self):
        """Set up the 3C3D test problem on CIFAR-10."""
        # Create dataset
        self.dataset = CIFAR10(batch_size=self._batch_size)

        # Create model
        self.model = ThreeC3D(num_outputs=10)
        self.model.to(self.device)

    def _compute_loss(self, outputs, targets, reduction='mean'):
        """Compute the cross-entropy loss.

        Args:
            outputs (torch.Tensor): Model outputs (logits) of shape (batch_size, 10).
            targets (torch.Tensor): Ground truth labels of shape (batch_size,) or
                (batch_size, 10) for one-hot encoded labels.
            reduction (str): 'mean' or 'none'. Defaults to 'mean'.

        Returns:
            torch.Tensor: Loss value(s).
        """
        # Handle one-hot encoded targets
        if targets.dim() == 2 and targets.size(1) > 1:
            targets = targets.argmax(dim=1)

        # Compute cross-entropy loss
        loss = F.cross_entropy(outputs, targets, reduction=reduction)

        return loss
