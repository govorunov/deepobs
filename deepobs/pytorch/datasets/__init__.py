"""DeepOBS datasets for PyTorch."""

from .dataset import DataSet
from .mnist import MNIST
from .fmnist import FashionMNIST
from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .svhn import SVHN
from .imagenet import ImageNet
from .tolstoi import Tolstoi
from .quadratic import Quadratic
from .two_d import TwoD

__all__ = [
    'DataSet',
    'MNIST',
    'FashionMNIST',
    'CIFAR10',
    'CIFAR100',
    'SVHN',
    'ImageNet',
    'Tolstoi',
    'Quadratic',
    'TwoD'
]
