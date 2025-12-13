"""3 Convolutional + 3 Dense architecture for PyTorch DeepOBS."""

import torch
import torch.nn as nn
import torch.nn.init as init


class ThreeC3D(nn.Module):
    """3 Convolutional + 3 Dense network for 32x32 RGB images.

    Architecture:
        - Conv1: 3 -> 64 filters, 5x5 kernel, valid padding, ReLU, MaxPool(3x3, stride=2)
        - Conv2: 64 -> 96 filters, 3x3 kernel, valid padding, ReLU, MaxPool(3x3, stride=2)
        - Conv3: 96 -> 128 filters, 3x3 kernel, same padding, ReLU, MaxPool(3x3, stride=2)
        - Flatten: 3x3x128 = 1152
        - FC1: 1152 -> 512, ReLU
        - FC2: 512 -> 256, ReLU
        - FC3: 256 -> num_outputs (no activation)

    Initialization:
        - Conv weights: Xavier/Glorot normal
        - FC weights: Xavier/Glorot uniform
        - All biases: constant 0.0

    Note: This architecture uses L2 regularization (weight_decay) by default.

    Args:
        num_outputs (int): Number of output units (classes).
    """

    def __init__(self, num_outputs):
        """Initialize the 3C3D model.

        Args:
            num_outputs (int): Number of output units.
        """
        super(ThreeC3D, self).__init__()

        # Convolutional layers
        # Conv1: 5x5 valid padding (no padding)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv2: 3x3 valid padding (no padding)
        self.conv2 = nn.Conv2d(64, 96, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv3: 3x3 same padding (padding=1)
        self.conv3 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Fully connected layers
        # Input calculation:
        # Start: 32x32
        # After conv1 (5x5, valid): 28x28
        # After pool1 (3x3, stride=2, padding=1): 14x14
        # After conv2 (3x3, valid): 12x12
        # After pool2 (3x3, stride=2, padding=1): 6x6
        # After conv3 (3x3, same): 6x6
        # After pool3 (3x3, stride=2, padding=1): 3x3
        self.fc1 = nn.Linear(3 * 3 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_outputs)

        # Initialize weights and biases (matching TensorFlow)
        # Conv layers: Xavier/Glorot normal
        for module in [self.conv1, self.conv2, self.conv3]:
            init.xavier_normal_(module.weight)
            init.constant_(module.bias, 0.0)

        # FC layers: Xavier/Glorot uniform
        for module in [self.fc1, self.fc2, self.fc3]:
            init.xavier_uniform_(module.weight)
            init.constant_(module.bias, 0.0)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

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

        # Third conv block
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(-1, 3 * 3 * 128)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
