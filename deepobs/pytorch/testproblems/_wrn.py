"""Wide Residual Network (WRN) architecture for PyTorch.

This module implements Wide ResNets based on the original paper:
"Wide Residual Networks" https://arxiv.org/abs/1605.07146

Key features:
- Residual connections with skip paths
- Batch normalization (BN -> ReLU -> Conv pre-activation pattern)
- Identity vs projection shortcuts
- Widening factor to increase network capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUnit(nn.Module):
    """A residual unit for Wide ResNet.

    Implements the pre-activation pattern: BN -> ReLU -> Conv

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the first convolution. Default: 1.
        bn_momentum (float): Momentum for batch normalization. Default: 0.1.
            Note: PyTorch momentum = 1 - TensorFlow momentum.
            TensorFlow default is 0.9, so PyTorch default is 0.1.
    """

    def __init__(self, in_channels, out_channels, stride=1, bn_momentum=0.1):
        super(ResidualUnit, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_momentum, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=1e-5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            # Use projection shortcut
            if in_channels == out_channels:
                # Same number of channels, just need to downsample
                self.shortcut = nn.MaxPool2d(kernel_size=stride, stride=stride)
            else:
                # Different number of channels, use 1x1 conv
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                         stride=stride, bias=False)
        else:
            # Identity shortcut
            self.shortcut = nn.Identity()

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after residual connection.
        """
        # Pre-activation: BN -> ReLU -> Conv
        out = self.bn1(x)
        out = F.relu(out)

        # Compute shortcut before first conv (uses pre-activated features)
        shortcut = self.shortcut(out)

        # First conv
        out = self.conv1(out)

        # Second block: BN -> ReLU -> Conv
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        # Add shortcut
        out = out + shortcut

        return out


class WideResNet(nn.Module):
    """Wide Residual Network.

    Args:
        num_residual_units (int): Number of residual units per block.
        widening_factor (int): Widening factor (k in the paper). Default: 4.
        num_outputs (int): Number of output classes. Default: 100.
        bn_momentum (float): Momentum for batch normalization. Default: 0.1.
            Note: PyTorch uses (1 - TensorFlow momentum).
    """

    def __init__(self, num_residual_units=6, widening_factor=4,
                 num_outputs=100, bn_momentum=0.1):
        super(WideResNet, self).__init__()

        self.num_residual_units = num_residual_units
        self.widening_factor = widening_factor
        self.num_outputs = num_outputs

        # Number of filters for each block
        filters = [16, 16 * widening_factor, 32 * widening_factor,
                   64 * widening_factor]
        # Strides for each block (first unit in each block)
        strides = [1, 2, 2]

        # Initial convolution
        self.conv_initial = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                                     padding=1, bias=False)

        # Build residual blocks
        self.block1 = self._make_block(16, filters[1], num_residual_units,
                                      strides[0], bn_momentum)
        self.block2 = self._make_block(filters[1], filters[2], num_residual_units,
                                      strides[1], bn_momentum)
        self.block3 = self._make_block(filters[2], filters[3], num_residual_units,
                                      strides[2], bn_momentum)

        # Final batch norm and FC
        self.bn_final = nn.BatchNorm2d(filters[3], momentum=bn_momentum, eps=1e-5)
        self.fc = nn.Linear(filters[3], num_outputs)

        # Initialize weights
        self._initialize_weights()

    def _make_block(self, in_channels, out_channels, num_units, stride, bn_momentum):
        """Build a residual block with multiple units.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_units (int): Number of residual units in this block.
            stride (int): Stride for the first unit (for downsampling).
            bn_momentum (float): Batch normalization momentum.

        Returns:
            nn.Sequential: The residual block.
        """
        layers = []

        # First unit (may downsample)
        layers.append(ResidualUnit(in_channels, out_channels, stride, bn_momentum))

        # Remaining units (no downsampling)
        for _ in range(1, num_units):
            layers.append(ResidualUnit(out_channels, out_channels, 1, bn_momentum))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_outputs).
        """
        # Initial convolution
        x = self.conv_initial(x)

        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Final BN and ReLU
        x = self.bn_final(x)
        x = F.relu(x)

        # Global average pooling
        x = torch.mean(x, dim=[2, 3])

        # Fully connected layer
        x = self.fc(x)

        return x


def wrn(num_residual_units=6, widening_factor=4, num_outputs=100, bn_momentum=0.1):
    """Create a Wide ResNet model.

    Args:
        num_residual_units (int): Number of residual units per block.
        widening_factor (int): Widening factor (k in the paper).
        num_outputs (int): Number of output classes.
        bn_momentum (float): Batch normalization momentum (PyTorch convention).

    Returns:
        WideResNet: WRN model instance.
    """
    return WideResNet(num_residual_units, widening_factor, num_outputs, bn_momentum)


def wrn_16_4(num_outputs=10):
    """WRN-16-4: 16 layers with widening factor 4.

    Total depth = 6n + 4 where n = num_residual_units
    For WRN-16-4: n = 2 (6*2 + 4 = 16)

    Args:
        num_outputs (int): Number of output classes.

    Returns:
        WideResNet: WRN-16-4 model instance.
    """
    return wrn(num_residual_units=2, widening_factor=4, num_outputs=num_outputs)


def wrn_40_4(num_outputs=100):
    """WRN-40-4: 40 layers with widening factor 4.

    Total depth = 6n + 4 where n = num_residual_units
    For WRN-40-4: n = 6 (6*6 + 4 = 40)

    Args:
        num_outputs (int): Number of output classes.

    Returns:
        WideResNet: WRN-40-4 model instance.
    """
    return wrn(num_residual_units=6, widening_factor=4, num_outputs=num_outputs)
