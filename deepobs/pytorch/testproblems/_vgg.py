"""VGG architecture for PyTorch.

This module implements VGG16 and VGG19 networks based on the original paper:
"Very Deep Convolutional Networks for Large-Scale Image Recognition"
https://arxiv.org/abs/1409.1556
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    """VGG network implementation.

    Args:
        variant (int): VGG variant (16 or 19).
        num_outputs (int): Number of output classes.
        input_size (tuple): Input image size (height, width). Default: (224, 224).
        batch_norm (bool): If True, adds BatchNorm after each conv layer. Default: False.
        dropout (float): Dropout probability for FC layers. Default: 0.5.
        bn_momentum (float): Momentum for BatchNorm layers. Default: 0.1.

    Note:
        - All conv filters are 3x3
        - ReLU activation after each conv
        - BatchNorm after conv (if enabled: Conv -> BN -> ReLU)
        - Max pooling (2x2) after each block
        - Dropout on FC layers (configurable)
        - Xavier/Glorot normal initialization
    """

    def __init__(self, variant=16, num_outputs=10, input_size=(224, 224),
                 batch_norm=False, dropout=0.5, bn_momentum=0.1):
        super(VGG, self).__init__()

        if variant not in [16, 19]:
            raise ValueError(f"VGG variant must be 16 or 19, got {variant}")

        self.variant = variant
        self.num_outputs = num_outputs
        self.input_size = input_size
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.bn_momentum = bn_momentum

        # Build convolutional layers
        self.features = self._make_layers()

        # Build classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(4096, num_outputs)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layers(self):
        """Build the convolutional feature extraction layers."""
        layers = []
        in_channels = 3

        # VGG16 configuration: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
        #                        512, 512, 512, 'M', 512, 512, 512, 'M']
        # VGG19 configuration: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
        #                        512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

        if self.variant == 16:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                   512, 512, 512, 'M', 512, 512, 512, 'M']
        else:  # variant == 19
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                   512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # Add conv layer (no bias if using BatchNorm)
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1,
                               bias=not self.batch_norm)
                layers.append(conv)

                # Add BatchNorm if enabled
                if self.batch_norm:
                    layers.append(nn.BatchNorm2d(v, momentum=self.bn_momentum, eps=1e-5))

                # Add ReLU activation
                layers.append(nn.ReLU(inplace=True))
                in_channels = v

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier/Glorot normal initialization
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm parameters
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # Xavier/Glorot normal initialization for linear layers
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_outputs).
        """
        # Resize to 224x224 if needed
        if x.shape[2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)

        # Feature extraction
        x = self.features(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x


def vgg16(num_outputs=10, input_size=(224, 224), batch_norm=False,
          dropout=0.5, bn_momentum=0.1):
    """Create a VGG16 model.

    Args:
        num_outputs (int): Number of output classes.
        input_size (tuple): Input image size (height, width).
        batch_norm (bool): If True, adds BatchNorm after conv layers.
        dropout (float): Dropout probability for FC layers.
        bn_momentum (float): Momentum for BatchNorm layers.

    Returns:
        VGG: VGG16 model instance.
    """
    return VGG(variant=16, num_outputs=num_outputs, input_size=input_size,
               batch_norm=batch_norm, dropout=dropout, bn_momentum=bn_momentum)


def vgg19(num_outputs=10, input_size=(224, 224), batch_norm=False,
          dropout=0.5, bn_momentum=0.1):
    """Create a VGG19 model.

    Args:
        num_outputs (int): Number of output classes.
        input_size (tuple): Input image size (height, width).
        batch_norm (bool): If True, adds BatchNorm after conv layers.
        dropout (float): Dropout probability for FC layers.
        bn_momentum (float): Momentum for BatchNorm layers.

    Returns:
        VGG: VGG19 model instance.
    """
    return VGG(variant=19, num_outputs=num_outputs, input_size=input_size,
               batch_norm=batch_norm, dropout=dropout, bn_momentum=bn_momentum)
