"""2 Conv + 2 Dense test problem on MNIST for PyTorch DeepOBS."""

import torch
import torch.nn.functional as F

from .testproblem import TestProblem
from ._2c2d import TwoC2D
from ..datasets.mnist import MNIST


class mnist_2c2d(TestProblem):
    """DeepOBS test problem class for a 2 conv + 2 dense network on MNIST.

    The network has:
        - Two convolutional layers with 32 and 64 filters (5x5 kernels)
        - Max pooling after each conv layer (2x2)
        - Two fully-connected layers with 1024 and 10 units
        - ReLU activations on all layers except the output
        - Truncated normal initialization (std=0.05) for weights
        - Constant initialization (0.05) for biases
        - Cross-entropy loss
        - No regularization

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float, optional): Weight decay (L2 regularization) factor.
            Not used for this test problem. Defaults to None.
        device (str or torch.device, optional): Device to use. Defaults to
            'cuda' if available, else 'cpu'.

    Attributes:
        dataset: The MNIST dataset instance.
        model: The 2C2D model (nn.Module).
        device: The device where computations are performed.
    """

    def __init__(self, batch_size, weight_decay=None, device=None):
        """Create a new 2C2D test problem on MNIST.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float, optional): Weight decay (not used). Defaults to None.
            device (str or torch.device, optional): Device to use.
        """
        super(mnist_2c2d, self).__init__(batch_size, weight_decay, device)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used "
                "for this test problem."
            )

    def set_up(self):
        """Set up the 2C2D test problem on MNIST."""
        # Create dataset
        self.dataset = MNIST(batch_size=self._batch_size)

        # Create model
        self.model = TwoC2D(num_outputs=10)
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
