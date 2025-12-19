"""DeepOBS datasets for PyTorch."""

from .dataset import DataSet
from .mnist import MNIST
from .fmnist import FashionMNIST
from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .svhn import SVHN
from .food101 import Food101
from .imagenet import ImageNet
from .penn_treebank import PennTreebank
from .quadratic import Quadratic
from .two_d import TwoD

__all__ = [
    'DataSet',
    'MNIST',
    'FashionMNIST',
    'CIFAR10',
    'CIFAR100',
    'SVHN',
    'Food101',
    'ImageNet',
    'PennTreebank',
    'Quadratic',
    'TwoD'
]
