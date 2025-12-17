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
from .cifar100_allcnnc import cifar100_allcnnc
from .cifar100_vgg16 import cifar100_vgg16
from .cifar100_vgg19 import cifar100_vgg19
from .cifar100_wrn404 import cifar100_wrn404

# CIFAR-10 VGG test problems
from .cifar10_vgg16 import cifar10_vgg16
from .cifar10_vgg19 import cifar10_vgg19

# SVHN test problems
from .svhn_3c3d import svhn_3c3d
from .svhn_wrn164 import svhn_wrn164

# ImageNet test problems
from .imagenet_vgg16 import imagenet_vgg16
from .imagenet_vgg19 import imagenet_vgg19
from .imagenet_inception_v3 import imagenet_inception_v3

# VAE test problems
from .mnist_vae import mnist_vae
from .fmnist_vae import fmnist_vae

# Text generation test problems
from .tolstoi_char_rnn import tolstoi_char_rnn
from .textgen import textgen

# Quadratic test problems
from .quadratic_deep import quadratic_deep

# 2D optimization test problems
from .two_d_rosenbrock import two_d_rosenbrock
from .two_d_beale import two_d_beale
from .two_d_branin import two_d_branin

__all__ = [
    'TestProblem',
    # MNIST
    'mnist_logreg',
    'mnist_mlp',
    'mnist_2c2d',
    'mnist_vae',
    # Fashion-MNIST
    'fmnist_logreg',
    'fmnist_mlp',
    'fmnist_2c2d',
    'fmnist_vae',
    # CIFAR-10
    'cifar10_3c3d',
    'cifar10_vgg16',
    'cifar10_vgg19',
    # CIFAR-100
    'cifar100_3c3d',
    'cifar100_allcnnc',
    'cifar100_vgg16',
    'cifar100_vgg19',
    'cifar100_wrn404',
    # SVHN
    'svhn_3c3d',
    'svhn_wrn164',
    # ImageNet
    'imagenet_vgg16',
    'imagenet_vgg19',
    'imagenet_inception_v3',
    # Text generation
    'tolstoi_char_rnn',
    'textgen',
    # Quadratic
    'quadratic_deep',
    # 2D optimization
    'two_d_rosenbrock',
    'two_d_beale',
    'two_d_branin',
]
