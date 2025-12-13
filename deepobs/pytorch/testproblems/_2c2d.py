"""2 Convolutional + 2 Dense architecture for PyTorch DeepOBS."""

import torch
import torch.nn as nn
import torch.nn.init as init


class TwoC2D(nn.Module):
    """2 Convolutional + 2 Dense network for 28x28 images.

    Architecture:
        - Conv1: 1 -> 32 filters, 5x5 kernel, padding=2, ReLU, MaxPool(2x2)
        - Conv2: 32 -> 64 filters, 5x5 kernel, padding=2, ReLU, MaxPool(2x2)
        - Flatten: 7x7x64 = 3136
        - FC1: 3136 -> 1024, ReLU
        - FC2: 1024 -> num_outputs (no activation)

    Initialization:
        - Weights: truncated normal with std=0.05
        - Biases: constant 0.05

    Args:
        num_outputs (int): Number of output units (classes).
    """

    def __init__(self, num_outputs):
        """Initialize the 2C2D model.

        Args:
            num_outputs (int): Number of output units.
        """
        super(TwoC2D, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        # After two 2x2 max pools: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, num_outputs)

        # Initialize weights and biases (matching TensorFlow)
        for module in [self.conv1, self.conv2, self.fc1, self.fc2]:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Truncated normal initialization with std=0.05
                init.trunc_normal_(module.weight, mean=0.0, std=0.05)
                init.constant_(module.bias, 0.05)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_outputs).
        """
        # First conv block
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # Second conv block
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # Flatten
        x = x.view(-1, 7 * 7 * 64)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x
