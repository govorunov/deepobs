"""A simple N-Dimensional Noisy Quadratic Problem with Deep Learning eigenvalues."""

import numpy as np
import torch
import torch.nn as nn

from .testproblem import TestProblem
from ..datasets.quadratic import Quadratic


def random_rotation(D, rng=None):
    """Produces a rotation matrix R in SO(D) (the special orthogonal
    group SO(D), or orthogonal matrices with unit determinant, drawn uniformly
    from the Haar measure.

    The algorithm used is the subgroup algorithm as originally proposed by
    P. Diaconis & M. Shahshahani, "The subgroup algorithm for generating
    uniform random variables". Probability in the Engineering and
    Informational Sciences 1: 15-32 (1987)

    Args:
        D (int): Dimensionality of the matrix.
        rng (np.random.RandomState, optional): Random state for reproducibility.

    Returns:
        np.array: Random rotation matrix ``R``.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    assert D >= 2
    D = int(D)  # make sure that the dimension is an integer

    # induction start: uniform draw from D=2 Haar measure
    t = 2 * np.pi * rng.uniform()
    R = [[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]

    for d in range(2, D):
        v = rng.normal(size=(d + 1, 1))
        # draw on S_d the unit sphere
        v = np.divide(v, np.sqrt(np.transpose(v).dot(v)))
        e = np.concatenate((np.array([[1.0]]), np.zeros((d, 1))), axis=0)
        # random coset location of SO(d-1) in SO(d)
        x = np.divide((e - v), (np.sqrt(np.transpose(e - v).dot(e - v))))

        D_mat = np.vstack([
            np.hstack([[[1.0]], np.zeros((1, d))]),
            np.hstack([np.zeros((d, 1)), R])
        ])
        R = D_mat - 2 * np.outer(x, np.transpose(x).dot(D_mat))
    # return negative to fix determinant
    return np.negative(R)


class QuadraticModel(nn.Module):
    """A simple parameter container for the quadratic problem.

    This is not a traditional neural network - it just holds a parameter
    vector theta that will be optimized against the quadratic objective.

    Args:
        dim (int): Dimensionality of the parameter vector.
    """

    def __init__(self, dim=100):
        super(QuadraticModel, self).__init__()
        # Initialize theta to 1.0 (as in TensorFlow version)
        self.theta = nn.Parameter(torch.ones(1, dim))

    def forward(self, x):
        """Forward pass just returns theta for all samples in the batch.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, dim).

        Returns:
            torch.Tensor: The theta parameter repeated for the batch.
        """
        # Return theta repeated for batch size
        batch_size = x.size(0)
        return self.theta.repeat(batch_size, 1)


class quadratic_deep(TestProblem):
    r"""DeepOBS test problem class for a stochastic quadratic test problem ``100``\
    dimensions. The 90% of the eigenvalues of the Hessian are drawn from the\
    interval :math:`(0.0, 1.0)` and the other 10% are from :math:`(30.0, 60.0)` \
    simulating an eigenspectrum which has been reported for Deep Learning \
    https://arxiv.org/abs/1611.01838.

    This creates a loss function of the form

    :math:`0.5 \cdot (\theta - x)^T \cdot Q \cdot (\theta - x)`

    with Hessian ``Q`` and "data" ``x`` coming from the quadratic data set, i.e.,
    zero-mean normal.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float, optional): No weight decay (L2-regularization) is used in this
            test problem. Defaults to ``None`` and any input here is ignored.
        device (str or torch.device, optional): Device to use. Defaults to
            'cuda' if available, else 'cpu'.

    Attributes:
        dataset: The DeepOBS data set class for the quadratic test problem.
        model: The parameter container (QuadraticModel).
        device: The device where computations are performed.
        hessian: The Hessian matrix Q of the quadratic function.
    """

    def __init__(self, batch_size, weight_decay=None, device=None):
        """Create a new quadratic deep test problem instance.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float, optional): No weight decay (L2-regularization) is used in this
                test problem. Defaults to ``None`` and any input here is ignored.
            device (str or torch.device, optional): Device to use.
        """
        super(quadratic_deep, self).__init__(batch_size, weight_decay, device)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used "
                "for this test problem."
            )

        # Generate the Hessian matrix (fixed for this test problem)
        rng = np.random.RandomState(42)
        eigenvalues = np.concatenate(
            (rng.uniform(0., 1., 90), rng.uniform(30., 60., 10)), axis=0)
        D = np.diag(eigenvalues)
        R = random_rotation(D.shape[0], rng)
        hessian = np.matmul(np.transpose(R), np.matmul(D, R))

        # Store as torch tensor
        self.hessian = torch.from_numpy(hessian).float()

    def set_up(self):
        """Set up the quadratic deep test problem."""
        # Create dataset
        self.dataset = Quadratic(batch_size=self._batch_size)

        # Create model (parameter container)
        self.model = QuadraticModel(dim=100)
        self.model.to(self.device)

        # Move hessian to device
        self.hessian = self.hessian.to(self.device)

    def _compute_loss(self, outputs, targets, reduction='mean'):
        """Compute the quadratic loss.

        The loss is: 0.5 * (theta - x)^T * Q * (theta - x)
        where theta is the model parameter, x is the data, and Q is the Hessian.

        Args:
            outputs (torch.Tensor): Model outputs (theta repeated for batch).
            targets (torch.Tensor): Data points x of shape (batch_size, 100).
            reduction (str): 'mean' or 'none'. Defaults to 'mean'.

        Returns:
            torch.Tensor: Loss value(s).
        """
        # outputs is theta repeated for batch, targets is x (the data)
        # Compute (theta - x)
        diff = outputs - targets  # Shape: (batch_size, 100)

        # Compute (theta - x)^T * Q * (theta - x) for each sample
        # This is: diff @ hessian @ diff.T, then take diagonal
        temp = torch.matmul(diff, self.hessian)  # (batch_size, 100)
        losses = 0.5 * torch.sum(temp * diff, dim=1)  # (batch_size,)

        if reduction == 'mean':
            return losses.mean()
        elif reduction == 'none':
            return losses
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}")

    def _compute_accuracy(self, outputs, targets):
        """Quadratic problems don't have accuracy.

        Args:
            outputs (torch.Tensor): Model outputs.
            targets (torch.Tensor): Targets.

        Returns:
            None: No accuracy for regression problems.
        """
        return None
