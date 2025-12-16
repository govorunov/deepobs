"""Inception V3 architecture for PyTorch.

This module implements Inception V3 based on the original paper:
"Rethinking the Inception Architecture for Computer Vision"
https://arxiv.org/abs/1512.00567

Key features:
- Multi-branch parallel convolutions
- Factorized convolutions (1x7, 7x1)
- Batch normalization with momentum 0.9997 (non-standard)
- No bias in conv layers (BN provides bias)
- Auxiliary classifier head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolutional layer followed by batch normalization and ReLU.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple): Stride of the convolution. Default: 1.
        padding (int, tuple, or str): Padding added to input. Default: 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()

        # Conv layer without bias (BN provides bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=False)

        # Batch norm with special momentum (TensorFlow uses 0.9997)
        # PyTorch momentum = 1 - TensorFlow momentum = 1 - 0.9997 = 0.0003
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.0003, eps=1e-3)

    def forward(self, x):
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class InceptionBlock5(nn.Module):
    """Inception block 5 (variants a and b).

    Args:
        in_channels (int): Number of input channels.
        variant (str): 'a' or 'b' variant.
    """

    def __init__(self, in_channels, variant='a'):
        super(InceptionBlock5, self).__init__()

        if variant == 'a':
            pool_filters = 32
        elif variant == 'b':
            pool_filters = 64
        else:
            raise ValueError('Variant must be "a" or "b"')

        # Branch 0: 1x1 conv
        self.branch0 = ConvBNReLU(in_channels, 64, kernel_size=1)

        # Branch 1: avg pool -> 1x1 conv
        self.branch1_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch1_conv = ConvBNReLU(in_channels, pool_filters, kernel_size=1)

        # Branch 2: 1x1 -> 5x5
        self.branch2_1x1 = ConvBNReLU(in_channels, 48, kernel_size=1)
        self.branch2_5x5 = ConvBNReLU(48, 64, kernel_size=5, padding=2)

        # Branch 3: 1x1 -> 3x3 -> 3x3
        self.branch3_1x1 = ConvBNReLU(in_channels, 64, kernel_size=1)
        self.branch3_3x3_0 = ConvBNReLU(64, 96, kernel_size=3, padding=1)
        self.branch3_3x3_1 = ConvBNReLU(96, 96, kernel_size=3, padding=1)

    def forward(self, x):
        """Forward pass."""
        branch0 = self.branch0(x)

        branch1 = self.branch1_pool(x)
        branch1 = self.branch1_conv(branch1)

        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_5x5(branch2)

        branch3 = self.branch3_1x1(x)
        branch3 = self.branch3_3x3_0(branch3)
        branch3 = self.branch3_3x3_1(branch3)

        outputs = torch.cat([branch0, branch1, branch2, branch3], dim=1)
        return outputs


class InceptionBlock10(nn.Module):
    """Inception block 10 (reduction block)."""

    def __init__(self, in_channels):
        super(InceptionBlock10, self).__init__()

        # Branch 0: max pool
        self.branch0_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Branch 1: 1x1 -> 3x3 (stride 2)
        self.branch1_3x3 = ConvBNReLU(in_channels, 384, kernel_size=3, stride=2, padding=0)

        # Branch 2: 1x1 -> 3x3 -> 3x3 (stride 2)
        self.branch2_1x1 = ConvBNReLU(in_channels, 64, kernel_size=1)
        self.branch2_3x3_0 = ConvBNReLU(64, 96, kernel_size=3, padding=1)
        self.branch2_3x3_1 = ConvBNReLU(96, 96, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        """Forward pass."""
        branch0 = self.branch0_pool(x)

        branch1 = self.branch1_3x3(x)

        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_3x3_0(branch2)
        branch2 = self.branch2_3x3_1(branch2)

        outputs = torch.cat([branch0, branch1, branch2], dim=1)
        return outputs


class InceptionBlock6(nn.Module):
    """Inception block 6 (variants a, b, c) with factorized convolutions.

    Args:
        in_channels (int): Number of input channels.
        variant (str): 'a', 'b', or 'c' variant.
    """

    def __init__(self, in_channels, variant='a'):
        super(InceptionBlock6, self).__init__()

        if variant == 'a':
            num_filters = 128
        elif variant == 'b':
            num_filters = 160
        elif variant == 'c':
            num_filters = 192
        else:
            raise ValueError('Variant must be "a", "b", or "c"')

        # Branch 0: 1x1 conv
        self.branch0 = ConvBNReLU(in_channels, 192, kernel_size=1)

        # Branch 1: avg pool -> 1x1 conv
        self.branch1_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch1_conv = ConvBNReLU(in_channels, 192, kernel_size=1)

        # Branch 2: 1x1 -> 1x7 -> 7x1
        self.branch2_1x1 = ConvBNReLU(in_channels, num_filters, kernel_size=1)
        self.branch2_1x7 = ConvBNReLU(num_filters, num_filters, kernel_size=(1, 7), padding=(0, 3))
        self.branch2_7x1 = ConvBNReLU(num_filters, 192, kernel_size=(7, 1), padding=(3, 0))

        # Branch 3: 1x1 -> 7x1 -> 1x7 -> 7x1 -> 1x7
        self.branch3_1x1 = ConvBNReLU(in_channels, num_filters, kernel_size=1)
        self.branch3_7x1_0 = ConvBNReLU(num_filters, num_filters, kernel_size=(7, 1), padding=(3, 0))
        self.branch3_1x7_0 = ConvBNReLU(num_filters, num_filters, kernel_size=(1, 7), padding=(0, 3))
        self.branch3_7x1_1 = ConvBNReLU(num_filters, num_filters, kernel_size=(7, 1), padding=(3, 0))
        self.branch3_1x7_1 = ConvBNReLU(num_filters, 192, kernel_size=(1, 7), padding=(0, 3))

    def forward(self, x):
        """Forward pass."""
        branch0 = self.branch0(x)

        branch1 = self.branch1_pool(x)
        branch1 = self.branch1_conv(branch1)

        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_1x7(branch2)
        branch2 = self.branch2_7x1(branch2)

        branch3 = self.branch3_1x1(x)
        branch3 = self.branch3_7x1_0(branch3)
        branch3 = self.branch3_1x7_0(branch3)
        branch3 = self.branch3_7x1_1(branch3)
        branch3 = self.branch3_1x7_1(branch3)

        outputs = torch.cat([branch0, branch1, branch2, branch3], dim=1)
        return outputs


class InceptionBlockD(nn.Module):
    """Inception block D (second reduction block)."""

    def __init__(self, in_channels):
        super(InceptionBlockD, self).__init__()

        # Branch 0: max pool
        self.branch0_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Branch 1: 1x1 -> 1x7 -> 7x1 -> 3x3 (stride 2)
        self.branch1_1x1 = ConvBNReLU(in_channels, 192, kernel_size=1)
        self.branch1_1x7 = ConvBNReLU(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch1_7x1 = ConvBNReLU(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch1_3x3 = ConvBNReLU(192, 192, kernel_size=3, stride=2, padding=0)

        # Branch 2: 1x1 -> 3x3 (stride 2)
        self.branch2_1x1 = ConvBNReLU(in_channels, 192, kernel_size=1)
        self.branch2_3x3 = ConvBNReLU(192, 320, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        """Forward pass."""
        branch0 = self.branch0_pool(x)

        branch1 = self.branch1_1x1(x)
        branch1 = self.branch1_1x7(branch1)
        branch1 = self.branch1_7x1(branch1)
        branch1 = self.branch1_3x3(branch1)

        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_3x3(branch2)

        outputs = torch.cat([branch0, branch1, branch2], dim=1)
        return outputs


class InceptionBlock7(nn.Module):
    """Inception block 7 (final blocks with expanded filter banks)."""

    def __init__(self, in_channels):
        super(InceptionBlock7, self).__init__()

        # Branch 0: 1x1 conv
        self.branch0 = ConvBNReLU(in_channels, 320, kernel_size=1)

        # Branch 1: 1x1 -> split into 1x3 and 3x1
        self.branch1_1x1 = ConvBNReLU(in_channels, 384, kernel_size=1)
        self.branch1_1x3 = ConvBNReLU(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch1_3x1 = ConvBNReLU(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # Branch 2: 1x1 -> 3x3 -> split into 1x3 and 3x1
        self.branch2_1x1 = ConvBNReLU(in_channels, 448, kernel_size=1)
        self.branch2_3x3 = ConvBNReLU(448, 384, kernel_size=3, padding=1)
        self.branch2_1x3 = ConvBNReLU(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch2_3x1 = ConvBNReLU(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # Branch 3: avg pool -> 1x1 conv
        self.branch3_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch3_conv = ConvBNReLU(in_channels, 192, kernel_size=1)

    def forward(self, x):
        """Forward pass."""
        branch0 = self.branch0(x)

        branch1 = self.branch1_1x1(x)
        branch1_1x3 = self.branch1_1x3(branch1)
        branch1_3x1 = self.branch1_3x1(branch1)
        branch1 = torch.cat([branch1_1x3, branch1_3x1], dim=1)

        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_3x3(branch2)
        branch2_1x3 = self.branch2_1x3(branch2)
        branch2_3x1 = self.branch2_3x1(branch2)
        branch2 = torch.cat([branch2_1x3, branch2_3x1], dim=1)

        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_conv(branch3)

        outputs = torch.cat([branch0, branch1, branch2, branch3], dim=1)
        return outputs


class InceptionV3(nn.Module):
    """Inception V3 network.

    Args:
        num_classes (int): Number of output classes. Default: 1001 (ImageNet + background).
        aux_logits (bool): Whether to use auxiliary classifier. Default: True.
    """

    def __init__(self, num_classes=1001, aux_logits=True):
        super(InceptionV3, self).__init__()

        self.num_classes = num_classes
        self.aux_logits = aux_logits

        # Stem
        self.conv1 = ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvBNReLU(32, 32, kernel_size=3, padding=0)
        self.conv3 = ConvBNReLU(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv4 = ConvBNReLU(64, 80, kernel_size=1, padding=0)
        self.conv5 = ConvBNReLU(80, 192, kernel_size=3, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Inception blocks
        self.block5a = InceptionBlock5(192, variant='a')
        self.block5b_0 = InceptionBlock5(256, variant='b')
        self.block5b_1 = InceptionBlock5(288, variant='b')

        self.block10 = InceptionBlock10(288)

        self.block6a = InceptionBlock6(768, variant='a')
        self.block6b_0 = InceptionBlock6(768, variant='b')
        self.block6b_1 = InceptionBlock6(768, variant='b')
        self.block6c = InceptionBlock6(768, variant='c')

        # Auxiliary classifier
        if self.aux_logits:
            self.aux_conv = ConvBNReLU(768, 128, kernel_size=1)
            self.aux_conv2 = ConvBNReLU(128, 768, kernel_size=5, padding=0)
            self.aux_fc = nn.Linear(768, num_classes)

        self.blockD = InceptionBlockD(768)

        self.block7_0 = InceptionBlock7(1280)
        self.block7_1 = InceptionBlock7(2048)

        # Output
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(2048, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor or tuple: If aux_logits=False, returns logits.
                If aux_logits=True, returns (logits, aux_logits).
        """
        # Resize to 299x299
        if x.shape[2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Stem
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)

        # Inception blocks
        x = self.block5a(x)
        x = self.block5b_0(x)
        x = self.block5b_1(x)

        x = self.block10(x)

        x = self.block6a(x)
        x = self.block6b_0(x)
        x = self.block6b_1(x)
        x = self.block6c(x)

        # Auxiliary classifier
        aux_out = None
        if self.aux_logits and self.training:
            aux = nn.functional.avg_pool2d(x, kernel_size=5, stride=3, padding=0)
            aux = self.aux_conv(aux)
            aux = self.aux_conv2(aux)
            aux = aux.view(aux.size(0), -1)
            aux_out = self.aux_fc(aux)

        x = self.blockD(x)

        x = self.block7_0(x)
        x = self.block7_1(x)

        # Output
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return x, aux_out
        else:
            return x


def inception_v3(num_classes=1001, aux_logits=True):
    """Create an Inception V3 model.

    Args:
        num_classes (int): Number of output classes.
        aux_logits (bool): Whether to use auxiliary classifier.

    Returns:
        InceptionV3: Inception V3 model instance.
    """
    return InceptionV3(num_classes=num_classes, aux_logits=aux_logits)
