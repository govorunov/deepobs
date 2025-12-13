"""Logistic regression test problem on Fashion-MNIST for PyTorch DeepOBS."""

import torch
import torch.nn.functional as F

from .testproblem import TestProblem
from ._logreg import LogisticRegression
from ..datasets.fmnist import FashionMNIST


class fmnist_logreg(TestProblem):
    """DeepOBS test problem class for logistic regression on Fashion-MNIST.

    The model is a single linear layer from flattened input (784) to 10 outputs.
    It uses:
        - Zero initialization for weights and biases
        - Cross-entropy loss
        - No regularization

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float, optional): Weight decay (L2 regularization) factor.
            Not used for this test problem. Defaults to None.
        device (str or torch.device, optional): Device to use. Defaults to
            'cuda' if available, else 'cpu'.

    Attributes:
        dataset: The Fashion-MNIST dataset instance.
        model: The logistic regression model (nn.Module).
        device: The device where computations are performed.
    """

    def __init__(self, batch_size, weight_decay=None, device=None):
        """Create a new logistic regression test problem on Fashion-MNIST.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float, optional): Weight decay (not used). Defaults to None.
            device (str or torch.device, optional): Device to use.
        """
        super(fmnist_logreg, self).__init__(batch_size, weight_decay, device)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used "
                "for this test problem."
            )

    def set_up(self):
        """Set up the logistic regression test problem on Fashion-MNIST."""
        # Create dataset
        self.dataset = FashionMNIST(batch_size=self._batch_size)

        # Create model
        self.model = LogisticRegression(num_outputs=10)
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
