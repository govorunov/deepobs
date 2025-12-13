# -*- coding: utf-8 -*-

# Try to import TensorFlow backend (may not be installed)
try:
    from . import tensorflow
except ImportError:
    tensorflow = None

# Try to import PyTorch backend (may not be installed)
try:
    from . import pytorch
except ImportError:
    pytorch = None

# Import analyzer and scripts (framework-agnostic, but may require dependencies)
try:
    from . import analyzer
except ImportError:
    analyzer = None

try:
    from . import scripts
except ImportError:
    scripts = None
