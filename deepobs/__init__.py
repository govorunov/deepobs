# -*- coding: utf-8 -*-

# Import PyTorch backend (may not be installed)
try:
    from . import pytorch
except ImportError:
    pytorch = None

# Import scripts (framework-agnostic, but may require dependencies)
try:
    from . import scripts
except ImportError:
    scripts = None
