"""Inception V3 architecture for ImageNet."""

import torch
import torch.nn.functional as F

from ._inception_v3 import inception_v3
from ..datasets.imagenet import ImageNet
from .testproblem import TestProblem


class imagenet_inception_v3(TestProblem):
    """DeepOBS test problem class for the Inception V3 network on ImageNet.

    Details about the architecture can be found in the original paper:
    "Rethinking the Inception Architecture for Computer Vision"
    https://arxiv.org/abs/1512.00567

    The Inception V3 model uses:
    - Multi-branch parallel convolutions
    - Factorized convolutions (1x7, 7x1)
    - Batch normalization with momentum 0.9997 (non-standard)
    - Auxiliary classifier head (used during training)
    - No bias in conv layers (BN provides bias)

    The model uses cross-entropy loss. A weight decay is used on the weights
    (but not the biases) which defaults to 5e-4.

    Note: ImageNet has 1000 classes, but the model outputs 1001 to include
    a background class.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
            is used on the weights but not the biases. Defaults to 5e-4.
        device (str or torch.device, optional): Device to use.
    """

    def __init__(self, batch_size, weight_decay=5e-4, device=None):
        """Create a new Inception V3 test problem instance on ImageNet.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float): Weight decay factor. Defaults to 5e-4.
            device (str or torch.device, optional): Device to use.
        """
        super(imagenet_inception_v3, self).__init__(batch_size, weight_decay, device)

    def set_up(self):
        """Set up the Inception V3 test problem on ImageNet."""
        # Initialize dataset
        self.dataset = ImageNet(self._batch_size)

        # Initialize model with auxiliary classifier
        # ImageNet has 1000 classes, but model outputs 1001 (includes background)
        self.model = inception_v3(num_classes=1001, aux_logits=True)
        self.model.to(self.device)

    def _compute_loss(self, outputs, targets, reduction='mean'):
        """Compute cross-entropy loss.

        For Inception V3, during training there are two outputs:
        - Main classifier output
        - Auxiliary classifier output (weighted at 0.3)

        Args:
            outputs (torch.Tensor or tuple): Model outputs. During training,
                this is a tuple (main_logits, aux_logits). During evaluation,
                just main_logits.
            targets (torch.Tensor): Ground truth labels (one-hot or class indices).
            reduction (str): 'mean' or 'none'.

        Returns:
            torch.Tensor: Loss value(s).
        """
        # Handle one-hot encoded targets
        if targets.dim() == 2 and targets.size(1) > 1:
            targets = targets.argmax(dim=1)

        # Check if we have auxiliary logits (training mode)
        if isinstance(outputs, tuple):
            main_logits, aux_logits = outputs

            # Main loss
            main_loss = F.cross_entropy(main_logits, targets, reduction=reduction)

            # Auxiliary loss (weighted at 0.3 in the original paper)
            aux_loss = F.cross_entropy(aux_logits, targets, reduction=reduction)

            # Combined loss
            if reduction == 'mean':
                total_loss = main_loss + 0.3 * aux_loss
            else:
                # For per-example losses, combine them
                total_loss = main_loss + 0.3 * aux_loss

            return total_loss
        else:
            # Evaluation mode (no auxiliary logits)
            return F.cross_entropy(outputs, targets, reduction=reduction)

    def _compute_accuracy(self, outputs, targets):
        """Compute accuracy.

        Args:
            outputs (torch.Tensor or tuple): Model outputs.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Accuracy (scalar).
        """
        # If we have auxiliary logits, only use main logits for accuracy
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Handle one-hot encoded targets
        if targets.dim() == 2 and targets.size(1) > 1:
            targets = targets.argmax(dim=1)

        # Get predictions
        predictions = outputs.argmax(dim=1)

        # Compute accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean()

        return accuracy
