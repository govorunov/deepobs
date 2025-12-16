"""The all convolutional model All-CNN-C for CIFAR-100."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets.cifar100 import cifar100
from .testproblem import TestProblem


class AllCNNC(nn.Module):
    """All Convolutional Neural Network C (All-CNN-C).

    Details about the architecture can be found in the original paper:
    https://arxiv.org/abs/1412.6806

    All-CNN-C uses only convolutional layers (no max pooling).
    Downsampling is achieved through strided convolutions.
    Progressive dropout is applied (0.2 -> 0.5).

    Args:
        num_outputs (int): Number of output classes.
    """

    def __init__(self, num_outputs=100):
        super(AllCNNC, self).__init__()

        self.num_outputs = num_outputs

        # First block (input dropout 0.2)
        self.dropout1 = nn.Dropout2d(0.2)
        self.conv1_1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)

        # Second block (dropout 0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.conv2_1 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)

        # Third block (dropout 0.5)
        self.dropout3 = nn.Dropout2d(0.5)
        self.conv3_1 = nn.Conv2d(192, 192, kernel_size=3, padding=0)  # valid padding
        self.conv3_2 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv3_3 = nn.Conv2d(192, num_outputs, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot normal and constant bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_outputs).
        """
        # First block with dropout 0.2
        x = self.dropout1(x)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))  # Strided conv for downsampling

        # Second block with dropout 0.5
        x = self.dropout2(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))  # Strided conv for downsampling

        # Third block with dropout 0.5
        x = self.dropout3(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))

        # Global average pooling
        x = torch.mean(x, dim=[2, 3])

        return x


class cifar100_allcnnc(TestProblem):
    """DeepOBS test problem class for the All Convolutional Neural Network C
    on Cifar-100.

    Details about the architecture can be found in the original paper:
    https://arxiv.org/abs/1412.6806

    The paper does not comment on initialization; here we use Xavier for conv
    filters and constant 0.1 for biases.

    A weight decay is used on the weights (but not the biases)
    which defaults to 5e-4.

    The reference training parameters from the paper are batch size = 256,
    num_epochs = 350 using the Momentum optimizer with mu = 0.9 and
    an initial learning rate of alpha = 0.05 and decrease by a factor of
    10 after 200, 250 and 300 epochs.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
            is used on the weights but not the biases. Defaults to 5e-4.
        device (str or torch.device, optional): Device to use.
    """

    def __init__(self, batch_size, weight_decay=0.0005, device=None):
        """Create a new All CNN C test problem instance on Cifar-100.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float): Weight decay factor. Defaults to 5e-4.
            device (str or torch.device, optional): Device to use.
        """
        super(cifar100_allcnnc, self).__init__(batch_size, weight_decay, device)

    def set_up(self):
        """Set up the All CNN C test problem on Cifar-100."""
        # Initialize dataset
        self.dataset = cifar100(self._batch_size)

        # Initialize model
        self.model = AllCNNC(num_outputs=100)
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
