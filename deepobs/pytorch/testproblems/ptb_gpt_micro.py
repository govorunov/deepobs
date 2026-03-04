"""A small decoder-only GPT for character-level language modelling on Penn Treebank."""

import torch.nn.functional as F

from .testproblem import TestProblem
from ._gpt_micro import GPTMicro
from ..datasets.penn_treebank import PennTreebank


class ptb_gpt_micro(TestProblem):
    """DeepOBS test problem class for GPT-Micro on Penn Treebank (character-level).

    A small decoder-only transformer for next-character prediction on Penn Treebank.
    The vocabulary is built from the dataset (same approach as ``ptb_lstm``),
    yielding a compact character-level vocabulary (~83 tokens).

    Network characteristics:
    - ``6`` transformer blocks
    - ``128``-dimensional model (d_model)
    - ``4`` attention heads
    - ``512``-dimensional feed-forward layer (d_ff)
    - sequence length ``128``
    - vocabulary derived from the dataset (same as ``ptb_lstm``)
    - weight-tied LM head
    - dropout ``0.1``
    - approximately ``1.22M`` parameters

    Working training parameters:
    - batch size ``64``
    - ``30-50`` epochs
    - Adam with a learning rate of approximately 3e-4 works

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float, optional): No weight decay (L2-regularization) is used in this
            test problem. Defaults to ``None`` and any input here is ignored.
        device (str or torch.device, optional): Device to use. Defaults to
            'cuda' if available, 'mps' on Apple Silicon, else 'cpu'.

    Attributes:
        dataset: The DeepOBS data set class for Penn Treebank (byte-level).
        model: The GPTMicro model (nn.Module).
        device: The device where computations are performed.
    """

    def __init__(self, batch_size, weight_decay=None, device=None):
        """Create a new GPT-Micro test problem instance on Penn Treebank.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float, optional): No weight decay (L2-regularization) is used in this
                test problem. Defaults to ``None`` and any input here is ignored.
            device (str or torch.device, optional): Device to use.
        """
        super(ptb_gpt_micro, self).__init__(batch_size, weight_decay, device)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used "
                "for this test problem."
            )

    def set_up(self):
        """Set up the GPT-Micro test problem instance on Penn Treebank."""
        self.dataset = PennTreebank(
            batch_size=self._batch_size,
            seq_length=128,
        )

        self.model = GPTMicro(vocab_size=self.dataset.vocab_size)
        self.model.to(self.device)

        print("GPT-Micro initialized:")
        print(f"  Vocabulary size: {self.dataset.vocab_size}")
        print(
            f"  Model parameters: ~{sum(p.numel() for p in self.model.parameters()):,}"
        )

    def _compute_loss(self, outputs, targets, reduction="mean"):
        """Compute the cross-entropy loss for sequence prediction.

        Args:
            outputs (torch.Tensor): Model logits of shape
                (batch_size, seq_length, vocab_size).
            targets (torch.Tensor): Ground truth token indices of shape
                (batch_size, seq_length).
            reduction (str): 'mean' or 'none'. Defaults to 'mean'.

        Returns:
            torch.Tensor: Loss value(s). If reduction='mean', returns a scalar.
                If reduction='none', returns per-example losses of shape (batch_size,).
        """
        batch_size, seq_length, vocab_size = outputs.size()

        outputs_flat = outputs.contiguous().view(-1, vocab_size)
        targets_flat = targets.contiguous().view(-1)

        token_losses = F.cross_entropy(
            outputs_flat, targets_flat, reduction="none"
        ).view(batch_size, seq_length)

        example_losses = token_losses.mean(dim=1)

        if reduction == "mean":
            return example_losses.mean()
        elif reduction == "none":
            return example_losses
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}")

    def _compute_accuracy(self, outputs, targets):
        """Compute per-token accuracy.

        Args:
            outputs (torch.Tensor): Model logits of shape
                (batch_size, seq_length, vocab_size).
            targets (torch.Tensor): Ground truth token indices of shape
                (batch_size, seq_length).

        Returns:
            torch.Tensor: Scalar accuracy (fraction of correctly predicted tokens).
        """
        predictions = outputs.argmax(dim=2)
        correct = (predictions == targets).float()
        return correct.mean()
