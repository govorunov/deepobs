"""A two-layer LSTM for character-level language modelling on Tolstoi's War and Peace."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .testproblem import TestProblem
from ..datasets.tolstoi import Tolstoi


class CharRNN(nn.Module):
    """A two-layer LSTM for character-level language modeling.

    Network characteristics:
    - Embedding layer (vocab_size -> 128)
    - 2-layer LSTM with 128 hidden units per layer
    - Dropout 0.2 on LSTM input and output (only during training)
    - Fully connected output layer (128 -> vocab_size)
    - Sequence length: 50
    - Stateful LSTM: hidden state persists across batches within an epoch

    Args:
        vocab_size (int): Size of the vocabulary (83 for War and Peace).
        embedding_dim (int): Dimension of character embeddings. Defaults to 128.
        hidden_size (int): Number of hidden units in LSTM layers. Defaults to 128.
        num_layers (int): Number of LSTM layers. Defaults to 2.
        dropout (float): Dropout probability. Defaults to 0.2.
    """

    def __init__(
        self,
        vocab_size=83,
        embedding_dim=128,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ):
        super(CharRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Dropout layers for input and output (mimicking DropoutWrapper)
        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Output fully connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Hidden state storage (will be set during forward pass)
        self.hidden = None

    def forward(self, x):
        """Forward pass through the character RNN.

        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_length, vocab_size).
        """
        batch_size, seq_length = x.size()

        # Embed the input
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)

        # Apply input dropout (only during training)
        if self.training:
            embedded = self.input_dropout(embedded)

        # LSTM forward pass with persistent state
        if self.hidden is not None:
            # Ensure hidden state has correct batch size
            if self.hidden[0].size(1) != batch_size:
                self.hidden = None

        lstm_out, new_hidden = self.lstm(embedded, self.hidden)
        # (batch_size, seq_length, hidden_size)

        # Update hidden state (detach to prevent backprop through time across batches)
        if self.training:
            self.hidden = tuple(h.detach() for h in new_hidden)
        else:
            # During evaluation, we also persist state but detach
            self.hidden = tuple(h.detach() for h in new_hidden)

        # Apply output dropout (only during training)
        if self.training:
            lstm_out = self.output_dropout(lstm_out)

        # Reshape for FC layer: (batch_size * seq_length, hidden_size)
        lstm_out_reshaped = lstm_out.contiguous().view(-1, self.hidden_size)

        # Apply output layer
        logits = self.fc(lstm_out_reshaped)  # (batch_size * seq_length, vocab_size)

        # Reshape back to (batch_size, seq_length, vocab_size)
        logits = logits.view(batch_size, seq_length, self.vocab_size)

        return logits

    def reset_hidden_state(self):
        """Reset the hidden state to None (zeros will be used on next forward)."""
        self.hidden = None


class tolstoi_char_rnn(TestProblem):
    """DeepOBS test problem class for a two-layer LSTM for character-level language
    modelling (Char RNN) on Tolstoi's War and Peace.

    Network characteristics:
    - ``128`` hidden units per LSTM cell
    - sequence length ``50``
    - cell state is automatically stored between batches within an epoch
    - cell state is reset to zero at the beginning of each epoch (train/eval)
    - dropout 0.2 (input_keep_prob=0.8, output_keep_prob=0.8 in TensorFlow terms)

    Working training parameters:
    - batch size ``50``
    - ``200`` epochs
    - SGD with a learning rate of approximately 0.1 works

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float, optional): No weight decay (L2-regularization) is used in this
            test problem. Defaults to ``None`` and any input here is ignored.
        device (str or torch.device, optional): Device to use. Defaults to
            'cuda' if available, else 'cpu'.

    Attributes:
        dataset: The DeepOBS data set class for Tolstoi.
        model: The CharRNN model (nn.Module).
        device: The device where computations are performed.
    """

    def __init__(self, batch_size, weight_decay=None, device=None):
        """Create a new Char RNN test problem instance on Tolstoi.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float, optional): No weight decay (L2-regularization) is used in this
                test problem. Defaults to ``None`` and any input here is ignored.
            device (str or torch.device, optional): Device to use.
        """
        super(tolstoi_char_rnn, self).__init__(batch_size, weight_decay, device)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used "
                "for this test problem."
            )

    def set_up(self):
        """Set up the Char RNN test problem instance on Tolstoi."""
        # Create dataset
        self.dataset = Tolstoi(batch_size=self._batch_size)

        # Create model
        # vocab_size=83 for War and Peace
        self.model = CharRNN(
            vocab_size=83,
            embedding_dim=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        self.model.to(self.device)

    def _compute_loss(self, outputs, targets, reduction='mean'):
        """Compute the cross-entropy loss for sequence prediction.

        The loss is computed per character position and averaged over the sequence.

        Args:
            outputs (torch.Tensor): Model outputs (logits) of shape
                (batch_size, seq_length, vocab_size).
            targets (torch.Tensor): Ground truth character indices of shape
                (batch_size, seq_length).
            reduction (str): 'mean' or 'none'. Defaults to 'mean'.

        Returns:
            torch.Tensor: Loss value(s). If reduction='mean', returns a scalar.
                If reduction='none', returns per-example losses of shape (batch_size,).
        """
        batch_size, seq_length, vocab_size = outputs.size()

        # Reshape outputs to (batch_size * seq_length, vocab_size)
        outputs_reshaped = outputs.contiguous().view(-1, vocab_size)

        # Reshape targets to (batch_size * seq_length)
        targets_reshaped = targets.contiguous().view(-1)

        # Compute cross-entropy loss per token
        # reduction='none' gives loss per token
        token_losses = F.cross_entropy(
            outputs_reshaped,
            targets_reshaped,
            reduction='none'
        )  # Shape: (batch_size * seq_length,)

        # Reshape to (batch_size, seq_length)
        token_losses = token_losses.view(batch_size, seq_length)

        # Average across time dimension (as in TensorFlow version with average_across_timesteps=True)
        # This gives per-example losses
        example_losses = token_losses.mean(dim=1)  # Shape: (batch_size,)

        if reduction == 'mean':
            return example_losses.mean()
        elif reduction == 'none':
            return example_losses
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}")

    def _compute_accuracy(self, outputs, targets):
        """Compute per-character accuracy.

        Args:
            outputs (torch.Tensor): Model outputs (logits) of shape
                (batch_size, seq_length, vocab_size).
            targets (torch.Tensor): Ground truth character indices of shape
                (batch_size, seq_length).

        Returns:
            torch.Tensor: Scalar accuracy (fraction of correctly predicted characters).
        """
        # Get predicted characters
        predictions = outputs.argmax(dim=2)  # (batch_size, seq_length)

        # Compare with targets
        correct = (predictions == targets).float()

        # Mean accuracy across all positions and examples
        accuracy = correct.mean()

        return accuracy

    def get_batch_loss_and_accuracy(self, batch, reduction='mean'):
        """Compute loss and accuracy for a batch.

        This method overrides the base class to handle LSTM state management.
        The hidden state is automatically managed by the model during forward pass.

        Args:
            batch (tuple): A tuple (inputs, targets) from a DataLoader.
            reduction (str): How to reduce the per-example losses. Options:
                - 'mean': Return mean loss (scalar)
                - 'none': Return per-example losses (vector)
                Defaults to 'mean'.

        Returns:
            tuple: (loss, accuracy) where:
                - loss is a torch.Tensor (scalar if reduction='mean', vector if 'none')
                - accuracy is a torch.Tensor (scalar)
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass (model manages hidden state internally)
        outputs = self.model(inputs)

        # Compute loss and accuracy
        loss = self._compute_loss(outputs, targets, reduction)
        accuracy = self._compute_accuracy(outputs, targets)

        return loss, accuracy

    def reset_state(self):
        """Reset the LSTM hidden state.

        This should be called at the beginning of each epoch or when switching
        between train/eval/test phases.
        """
        if self.model is not None:
            self.model.reset_hidden_state()
