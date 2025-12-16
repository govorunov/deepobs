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

    Note:
        - All conv filters are 3x3
        - ReLU activation after each conv
        - Max pooling (2x2) after each block
        - Dropout 0.5 on FC layers
        - Xavier/Glorot normal initialization
    """

    def __init__(self, variant=16, num_outputs=10, input_size=(224, 224)):
        super(VGG, self).__init__()

        if variant not in [16, 19]:
            raise ValueError(f"VGG variant must be 16 or 19, got {variant}")

        self.variant = variant
        self.num_outputs = num_outputs
        self.input_size = input_size

        # Build convolutional layers
        self.features = self._make_layers()

        # Build classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
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
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
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


def vgg16(num_outputs=10, input_size=(224, 224)):
    """Create a VGG16 model.

    Args:
        num_outputs (int): Number of output classes.
        input_size (tuple): Input image size (height, width).

    Returns:
        VGG: VGG16 model instance.
    """
    return VGG(variant=16, num_outputs=num_outputs, input_size=input_size)


def vgg19(num_outputs=10, input_size=(224, 224)):
    """Create a VGG19 model.

    Args:
        num_outputs (int): Number of output classes.
        input_size (tuple): Input image size (height, width).

    Returns:
        VGG: VGG19 model instance.
    """
    return VGG(variant=19, num_outputs=num_outputs, input_size=input_size)
