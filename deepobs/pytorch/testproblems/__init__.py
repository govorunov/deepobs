"""DeepOBS test problems for PyTorch."""

from .testproblem import TestProblem

# MNIST test problems
from .mnist_logreg import mnist_logreg
from .mnist_mlp import mnist_mlp
from .mnist_2c2d import mnist_2c2d

# Fashion-MNIST test problems
from .fmnist_logreg import fmnist_logreg
from .fmnist_mlp import fmnist_mlp
from .fmnist_2c2d import fmnist_2c2d

# CIFAR-10 test problems
from .cifar10_3c3d import cifar10_3c3d

# CIFAR-100 test problems
from .cifar100_3c3d import cifar100_3c3d

__all__ = [
    'TestProblem',
    # MNIST
    'mnist_logreg',
    'mnist_mlp',
    'mnist_2c2d',
    # Fashion-MNIST
    'fmnist_logreg',
    'fmnist_mlp',
    'fmnist_2c2d',
    # CIFAR-10
    'cifar10_3c3d',
    # CIFAR-100
    'cifar100_3c3d',
]
