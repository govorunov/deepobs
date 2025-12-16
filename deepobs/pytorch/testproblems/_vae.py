"""Variational Autoencoder (VAE) architecture for PyTorch.

This module implements a VAE with convolutional encoder and decoder,
designed for 28x28 grayscale images (MNIST, Fashion-MNIST).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    """Encoder part of the VAE.

    The encoder consists of three convolutional layers followed by two dense
    layers that output the mean and log standard deviation of the latent
    distribution.

    Args:
        n_latent (int): Size of the latent space. Default: 8.
    """

    def __init__(self, n_latent=8):
        super(VAEEncoder, self).__init__()

        self.n_latent = n_latent

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.dropout2 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=1)
        self.dropout3 = nn.Dropout2d(0.2)

        # Dense layers for mean and log std
        # After convolutions: 28x28 -> 14x14 -> 7x7 -> 7x7
        self.fc_mean = nn.Linear(64 * 7 * 7, n_latent)
        self.fc_log_std = nn.Linear(64 * 7 * 7, n_latent)

    def forward(self, x):
        """Forward pass through encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 784) or (batch_size, 1, 28, 28).

        Returns:
            tuple: (z, mean, log_std) where:
                - z: Sampled latent vector (batch_size, n_latent)
                - mean: Mean of latent distribution (batch_size, n_latent)
                - log_std: Log standard deviation (batch_size, n_latent)
        """
        # Reshape if needed
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        # Convolutional layers with leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.3)
        x = self.dropout1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=0.3)
        x = self.dropout2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=0.3)
        x = self.dropout3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Compute mean and log_std
        mean = self.fc_mean(x)
        log_std = 0.5 * self.fc_log_std(x)  # Match TensorFlow: 0.5 * dense output

        # Reparameterization trick: z = mean + epsilon * exp(log_std)
        epsilon = torch.randn_like(mean)
        z = mean + epsilon * torch.exp(log_std)

        return z, mean, log_std


class VAEDecoder(nn.Module):
    """Decoder part of the VAE.

    The decoder takes a latent vector and reconstructs it back to the original
    image space using dense and transposed convolutional layers.

    Args:
        n_latent (int): Size of the latent space. Default: 8.
    """

    def __init__(self, n_latent=8):
        super(VAEDecoder, self).__init__()

        self.n_latent = n_latent

        # Dense layers
        self.fc1 = nn.Linear(n_latent, 24)
        self.fc2 = nn.Linear(24, 24 * 2 + 1)  # 49

        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.dropout1 = nn.Dropout2d(0.2)

        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=1)
        self.dropout2 = nn.Dropout2d(0.2)

        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=1)

        # Final dense layer to image
        self.fc_out = nn.Linear(64 * 14 * 14, 28 * 28)

    def forward(self, z):
        """Forward pass through decoder.

        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, n_latent).

        Returns:
            torch.Tensor: Reconstructed image of shape (batch_size, 28, 28).
        """
        # Dense layers with leaky ReLU
        x = F.leaky_relu(self.fc1(z), negative_slope=0.3)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.3)

        # Reshape to 7x7x1
        x = x.view(-1, 1, 7, 7)

        # Deconvolutional layers with ReLU
        x = F.relu(self.deconv1(x))  # 7x7 -> 14x14
        x = self.dropout1(x)

        x = F.relu(self.deconv2(x))  # 14x14 -> 14x14
        x = self.dropout2(x)

        x = self.deconv3(x)  # 14x14 -> 14x14

        # Flatten and final dense layer with sigmoid
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc_out(x))

        # Reshape to image
        img = x.view(-1, 28, 28)

        return img


class VAE(nn.Module):
    """Variational Autoencoder (VAE).

    Combines encoder and decoder into a single model.

    Args:
        n_latent (int): Size of the latent space. Default: 8.
    """

    def __init__(self, n_latent=8):
        super(VAE, self).__init__()

        self.n_latent = n_latent
        self.encoder = VAEEncoder(n_latent)
        self.decoder = VAEDecoder(n_latent)

    def forward(self, x):
        """Forward pass through VAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 784) or (batch_size, 1, 28, 28).

        Returns:
            tuple: (reconstruction, mean, log_std) where:
                - reconstruction: Reconstructed image (batch_size, 28, 28)
                - mean: Mean of latent distribution (batch_size, n_latent)
                - log_std: Log standard deviation (batch_size, n_latent)
        """
        z, mean, log_std = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, mean, log_std


def vae(n_latent=8):
    """Create a VAE model.

    Args:
        n_latent (int): Size of the latent space. Default: 8.

    Returns:
        VAE: VAE model instance.
    """
    return VAE(n_latent=n_latent)
