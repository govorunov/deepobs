"""A simple 2D Noisy Branin Loss Function."""

import numpy as np
import torch
import torch.nn as nn

from .testproblem import TestProblem
from ..datasets.two_d import TwoD


class BraninModel(nn.Module):
    """A simple parameter container for the 2D Branin function.

    This is not a traditional neural network - it just holds two scalar
    parameters (u, v) that will be optimized against the Branin objective.

    Args:
        starting_point (list): Initial values for [u, v].
    """

    def __init__(self, starting_point=None):
        super(BraninModel, self).__init__()
        if starting_point is None:
            starting_point = [2.5, 12.5]

        # Create scalar parameters u and v
        self.u = nn.Parameter(torch.tensor(starting_point[0]))
        self.v = nn.Parameter(torch.tensor(starting_point[1]))

    def forward(self, x):
        """Forward pass returns the parameters u and v.

        Args:
            x (torch.Tensor): Not used, but needed for interface consistency.

        Returns:
            tuple: (u, v) parameters.
        """
        return self.u, self.v


class two_d_branin(TestProblem):
    r"""DeepOBS test problem class for a stochastic version of the\
    two-dimensional Branin function as the loss function.

    Using the deterministic `Branin function
    <https://www.sfu.ca/~ssurjano/branin.html>`_ and adding stochastic noise of
    the form

    :math:`u \cdot x + v \cdot y`

    where ``x`` and ``y`` are normally distributed with mean ``0.0`` and
    standard deviation ``1.0`` we get a loss function of the form

    :math:`(v - 5.1/(4 \cdot \pi^2) u^2 + 5/ \pi u - 6)^2 +\
    10 \cdot (1-1/(8 \cdot \pi)) \cdot \cos(u) + 10 + u \cdot x + v \cdot y`.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float, optional): No weight decay (L2-regularization) is used in this
            test problem. Defaults to ``None`` and any input here is ignored.
        device (str or torch.device, optional): Device to use. Defaults to
            'cuda' if available, else 'cpu'.

    Attributes:
        dataset: The DeepOBS data set class for the two_d stochastic test problem.
        model: The parameter container (BraninModel).
        device: The device where computations are performed.
    """

    def __init__(self, batch_size, weight_decay=None, device=None):
        """Create a new 2D Branin test problem instance.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float, optional): No weight decay (L2-regularization) is used in this
                test problem. Defaults to ``None`` and any input here is ignored.
            device (str or torch.device, optional): Device to use.
        """
        super(two_d_branin, self).__init__(batch_size, weight_decay, device)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used "
                "for this test problem."
            )

    def set_up(self):
        """Sets up the stochastic two-dimensional Branin test problem.
        Using ``2.5`` and ``12.5`` as a starting point for the weights ``u``
        and ``v``.
        """
        # Create dataset
        self.dataset = TwoD(batch_size=self._batch_size)

        # Create model with starting point [2.5, 12.5]
        self.model = BraninModel(starting_point=[2.5, 12.5])
        self.model.to(self.device)

    def _compute_loss(self, outputs, targets, reduction='mean'):
        """Compute the Branin loss.

        The loss is: a*(v - b*u^2 + c*u - r)^2 + s*(1 - t)*cos(u) + s + u*x + v*y
        where a=1, b=5.1/(4*pi^2), c=5/pi, r=6, s=10, t=1/(8*pi)

        Args:
            outputs (tuple): Model outputs (u, v).
            targets (torch.Tensor): Data points (x, y) of shape (batch_size, 2).
            reduction (str): 'mean' or 'none'. Defaults to 'mean'.

        Returns:
            torch.Tensor: Loss value(s).
        """
        u, v = outputs

        # Extract x and y from targets
        x = targets[:, 0]  # First column
        y = targets[:, 1]  # Second column

        # Define Branin constants
        a = 1.
        b = 5.1 / (4. * np.pi**2)
        c = 5 / np.pi
        r = 6.
        s = 10.
        t = 1 / (8. * np.pi)

        # Compute Branin function
        losses = (a * (v - b * u**2 + c * u - r)**2 +
                  s * (1 - t) * torch.cos(u) + s +
                  u * x + v * y)

        if reduction == 'mean':
            return losses.mean()
        elif reduction == 'none':
            return losses
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}")

    def _compute_accuracy(self, outputs, targets):
        """2D optimization problems don't have accuracy.

        Args:
            outputs: Model outputs.
            targets: Targets.

        Returns:
            None: No accuracy for optimization problems.
        """
        return None
