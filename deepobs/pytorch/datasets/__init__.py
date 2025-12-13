"""DeepOBS datasets for PyTorch."""

from .dataset import DataSet
from .mnist import MNIST
from .fmnist import FashionMNIST
from .cifar10 import CIFAR10
from .cifar100 import CIFAR100

__all__ = [
    'DataSet',
    'MNIST',
    'FashionMNIST',
    'CIFAR10',
    'CIFAR100'
]
