"""Wide ResNet 40-4 architecture for CIFAR-100."""

import torch
import torch.nn.functional as F

from ._wrn import wrn_40_4
from ..datasets.cifar100 import CIFAR100
from .testproblem import TestProblem


class cifar100_wrn404(TestProblem):
    """DeepOBS test problem class for the Wide Residual Network 40-4 architecture
    for CIFAR-100.

    Details about the architecture can be found in the original paper:
    https://arxiv.org/abs/1605.07146

    A weight decay is used on the weights (but not the biases)
    which defaults to 5e-4.

    Training settings recommended in the original paper:
    batch size = 128, num_epochs = 200 using the Momentum optimizer
    with mu = 0.9 and an initial learning rate of 0.1 with a decrease by
    0.2 after 60, 120 and 160 epochs.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
            is used on the weights but not the biases. Defaults to 5e-4.
        device (str or torch.device, optional): Device to use.
    """

    def __init__(self, batch_size, weight_decay=0.0005, device=None):
        """Create a new WRN 40-4 test problem instance on Cifar-100.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float): Weight decay factor. Defaults to 5e-4.
            device (str or torch.device, optional): Device to use.
        """
        super(cifar100_wrn404, self).__init__(batch_size, weight_decay, device)

    def set_up(self):
        """Set up the Wide ResNet 40-4 test problem on Cifar-100."""
        # Initialize dataset
        self.dataset = CIFAR100(self._batch_size)

        # Initialize model
        # Note: TensorFlow uses bn_momentum=0.9, PyTorch uses 1 - 0.9 = 0.1
        self.model = wrn_40_4(num_outputs=100)
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
