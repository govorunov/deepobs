"""Logistic Regression architecture for PyTorch DeepOBS."""

import torch.nn as nn


class LogisticRegression(nn.Module):
    """Logistic regression model for 28x28 images.

    A single linear layer from flattened input (784) to num_outputs.
    Weights and biases are initialized to 0.0.

    Args:
        num_outputs (int): Number of output units (classes).
    """

    def __init__(self, num_outputs):
        """Initialize the logistic regression model.

        Args:
            num_outputs (int): Number of output units.
        """
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(784, num_outputs)

        # Initialize weights and biases to 0.0 (matching TensorFlow)
        nn.init.constant_(self.fc.weight, 0.0)
        nn.init.constant_(self.fc.bias, 0.0)

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
        return self.fc(x)
