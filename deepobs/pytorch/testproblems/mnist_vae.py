"""A Variational Autoencoder architecture for MNIST."""

import torch
import torch.nn.functional as F

from ._vae import vae
from ..datasets.mnist import MNIST
from .testproblem import TestProblem


class mnist_vae(TestProblem):
    """DeepOBS test problem class for a variational autoencoder (VAE) on MNIST.

    The network consists of an encoder:
      - With three convolutional layers with each 64 filters.
      - Using a leaky ReLU activation function with alpha = 0.3
      - Dropout layers after each convolutional layer with a rate of 0.2.

    and a decoder:
      - With two dense layers with 24 and 49 units and leaky ReLU activation.
      - With three deconvolutional layers with each 64 filters.
      - Dropout layers after the first two deconvolutional layer with a rate of 0.2.
      - A final dense layer with 28 x 28 units and sigmoid activation.

    No regularization is used.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float): No weight decay (L2-regularization) is used in this
            test problem. Defaults to None and any input here is ignored.
        device (str or torch.device, optional): Device to use.
    """

    def __init__(self, batch_size, weight_decay=None, device=None):
        """Create a new VAE test problem instance on MNIST.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float): No weight decay is used. Defaults to None.
            device (str or torch.device, optional): Device to use.
        """
        super(mnist_vae, self).__init__(batch_size, weight_decay, device)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used "
                "for this model."
            )

    def set_up(self):
        """Sets up the VAE test problem on MNIST."""
        # Initialize dataset
        self.dataset = MNIST(self._batch_size)

        # Initialize model
        self.model = vae(n_latent=8)
        self.model.to(self.device)

    def _compute_loss(self, outputs, targets, reduction='mean'):
        """Compute VAE loss (reconstruction + KL divergence).

        The VAE loss consists of two components:
        1. Reconstruction loss: Measures how well the decoder reconstructs the input
        2. KL divergence: Regularizes the latent space to be close to N(0, 1)

        Args:
            outputs (tuple): Model outputs (reconstruction, mean, log_std).
            targets (torch.Tensor): Ground truth images (not used, VAE is unsupervised).
            reduction (str): 'mean' or 'none'.

        Returns:
            torch.Tensor: Loss value(s).
        """
        # Unpack VAE outputs
        reconstruction, mean, log_std = outputs

        # Get the input images (we need them for reconstruction loss)
        # In VAE, we're comparing reconstruction to input, not to targets
        # The input is stored in self._last_input during get_batch_loss_and_accuracy
        # For simplicity, we'll compute it here using targets as input proxy
        # Note: In practice, the runner should handle this properly

        # Reconstruction loss: Mean Squared Error
        # TensorFlow version uses: tf.reduce_sum(tf.squared_difference(flatten_img, x_flat), 1)
        flatten_reconstruction = reconstruction.view(-1, 28 * 28)
        x_flat = targets.view(-1, 28 * 28) if targets.dim() > 2 else targets

        # Note: We're using the input (which should be the same as targets for VAE)
        reconstruction_loss = torch.sum((flatten_reconstruction - x_flat) ** 2, dim=1)

        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, 1)
        # Formula: -0.5 * sum(1 + 2*log_std - mean^2 - exp(2*log_std))
        kl_loss = -0.5 * torch.sum(
            1.0 + 2.0 * log_std - mean ** 2 - torch.exp(2.0 * log_std),
            dim=1
        )

        # Total loss
        total_loss = reconstruction_loss + kl_loss

        if reduction == 'mean':
            return total_loss.mean()
        else:
            return total_loss

    def get_batch_loss_and_accuracy(self, batch, reduction='mean'):
        """Compute loss for a batch.

        VAE doesn't have accuracy metric, so we return None for accuracy.

        Args:
            batch (tuple): A tuple (inputs, targets) from a DataLoader.
            reduction (str): How to reduce the per-example losses.

        Returns:
            tuple: (loss, None) - accuracy is not applicable for VAE.
        """
        inputs, _ = batch  # Ignore targets, VAE is unsupervised
        inputs = inputs.to(self.device)

        # Forward pass
        reconstruction, mean, log_std = self.model(inputs)

        # Compute loss using inputs as "targets"
        loss = self._compute_loss((reconstruction, mean, log_std), inputs, reduction)

        # No accuracy for VAE
        return loss, None

    def _compute_accuracy(self, outputs, targets):
        """VAE doesn't have accuracy metric.

        Returns:
            None
        """
        return None
