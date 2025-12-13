"""Multi-Layer Perceptron architecture for PyTorch DeepOBS."""

import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    """Multi-layer perceptron for 28x28 images.

    Architecture:
        - Input: 784 (flattened 28x28 image)
        - FC1: 784 -> 1000, ReLU
        - FC2: 1000 -> 500, ReLU
        - FC3: 500 -> 100, ReLU
        - FC4: 100 -> num_outputs (no activation)

    Initialization:
        - Weights: truncated normal with std=0.03
        - Biases: constant 0.0

    Args:
        num_outputs (int): Number of output units (classes).
    """

    def __init__(self, num_outputs):
        """Initialize the MLP model.

        Args:
            num_outputs (int): Number of output units.
        """
        super(MLP, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, num_outputs)

        # Initialize weights and biases (matching TensorFlow)
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            # Truncated normal initialization with std=0.03
            init.trunc_normal_(layer.weight, mean=0.0, std=0.03)
            init.constant_(layer.bias, 0.0)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
                or (batch_size, 28, 28).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_outputs).
        """
        # Flatten input
        x = x.view(-1, 784)

        # Forward through layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x


# Import torch for relu
import torch
