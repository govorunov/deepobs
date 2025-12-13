"""Base class for DeepOBS test problems."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestProblem(ABC):
    """Base class for DeepOBS test problems.

    A test problem combines a dataset with a model architecture and defines
    how to compute loss and accuracy. This class provides a unified interface
    for training and evaluating models.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float, optional): Weight decay (L2-regularization) factor.
            If not specified, test problems use their default values. Note that
            some test problems do not use regularization.
        device (str or torch.device, optional): Device to use ('cpu', 'cuda', or
            a torch.device object). Defaults to 'cuda' if available, else 'cpu'.

    Attributes:
        dataset: The DeepOBS dataset instance (datasets.DataSet).
        model (nn.Module): The PyTorch model/neural network.
        device (torch.device): The device where computations are performed.
    """

    def __init__(
        self,
        batch_size: int,
        weight_decay: Optional[float] = None,
        device: Optional[str] = None
    ):
        """Creates a new test problem instance.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float, optional): Weight decay factor for L2 regularization.
            device (str or torch.device, optional): Device for computations.
        """
        self._batch_size = batch_size
        self._weight_decay = weight_decay

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Public attributes (set by subclasses in set_up method)
        self.dataset = None
        self.model = None

    @property
    def batch_size(self) -> int:
        """The batch size used by this test problem."""
        return self._batch_size

    @property
    def weight_decay(self) -> Optional[float]:
        """The weight decay (L2 regularization) factor."""
        return self._weight_decay

    @abstractmethod
    def set_up(self) -> None:
        """Sets up the test problem.

        This method must:
        1. Create self.dataset (a DataSet instance)
        2. Create self.model (a nn.Module instance)
        3. Move model to self.device

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            "'TestProblem' is an abstract base class. "
            "Subclasses must implement 'set_up'."
        )

    def get_batch_loss_and_accuracy(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute loss and accuracy for a batch.

        Args:
            batch (tuple): A tuple (inputs, targets) from a DataLoader.
            reduction (str): How to reduce the per-example losses. Options:
                - 'mean': Return mean loss (scalar)
                - 'none': Return per-example losses (vector)
                Defaults to 'mean'.

        Returns:
            tuple: (loss, accuracy) where:
                - loss is a torch.Tensor (scalar if reduction='mean', vector if 'none')
                - accuracy is a torch.Tensor (scalar) or None if not applicable
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        outputs = self.model(inputs)

        # Compute loss (implemented by subclass)
        loss = self._compute_loss(outputs, targets, reduction)

        # Compute accuracy (if applicable)
        accuracy = self._compute_accuracy(outputs, targets)

        return loss, accuracy

    @abstractmethod
    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute loss for a batch.

        Args:
            outputs (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Ground truth labels.
            reduction (str): 'mean' or 'none'.

        Returns:
            torch.Tensor: Loss value(s).
        """
        raise NotImplementedError(
            "Subclasses must implement '_compute_loss'."
        )

    def _compute_accuracy(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Compute accuracy for a batch.

        Default implementation for classification tasks. Subclasses can override
        if a different accuracy metric is needed.

        Args:
            outputs (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor or None: Accuracy (scalar) or None if not applicable.
        """
        # Default: classification accuracy
        # Handle one-hot targets (convert to class indices)
        if targets.dim() == 2 and targets.size(1) > 1:
            targets = targets.argmax(dim=1)

        # Get predictions
        predictions = outputs.argmax(dim=1)

        # Compute accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean()

        return accuracy

    def get_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss (L2 penalty on weights).

        Note: In PyTorch, it's generally better to use the optimizer's
        weight_decay parameter instead of manually adding L2 loss. This
        method is provided for compatibility and special cases.

        Returns:
            torch.Tensor: Scalar regularization loss (or 0.0 if no regularization).
        """
        if self._weight_decay is None or self._weight_decay == 0.0:
            return torch.tensor(0.0, device=self.device)

        # Compute L2 penalty on all model parameters
        l2_loss = torch.tensor(0.0, device=self.device)
        for param in self.model.parameters():
            l2_loss += torch.sum(param ** 2)

        return 0.5 * self._weight_decay * l2_loss
