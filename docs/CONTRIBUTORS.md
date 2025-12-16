# DeepOBS Contributors

**Project**: DeepOBS - Deep Learning Optimizer Benchmark Suite
**License**: MIT
**Version**: 1.2.0 (PyTorch Support)

---

## Original Authors

### Frank Schneider
**Role**: Project Lead, Original Author
**Affiliation**: Max Planck Institute for Intelligent Systems, Tübingen
**Contribution**:
- Conceived and designed DeepOBS
- Led original TensorFlow implementation
- Co-authored ICLR 2019 paper
- Project maintenance and oversight

**Contact**: frank.schneider@tue.mpg.de

### Lukas Balles
**Role**: Co-Author
**Affiliation**: Max Planck Institute for Intelligent Systems, Tübingen
**Contribution**:
- Co-designed DeepOBS architecture
- Contributed to TensorFlow implementation
- Co-authored ICLR 2019 paper
- Baseline experiments and analysis

### Philipp Hennig
**Role**: Co-Author, Research Supervisor
**Affiliation**: Max Planck Institute for Intelligent Systems, Tübingen
**Contribution**:
- Research supervision and guidance
- Co-authored ICLR 2019 paper
- Strategic direction and methodology

---

## PyTorch Implementation Team

### Aaron Bahde
**Role**: DeepOBS 1.2.0 Development Lead
**Contribution**:
- Led initial PyTorch implementation efforts
- Developed pre-release version (1.2.0-beta0)
- Interface improvements and bug fixes
- Community engagement

**Note**: Many thanks to Aaron for spearheading the development of DeepOBS 1.2.0 and laying the groundwork for the PyTorch implementation.

### PyTorch Migration Team (2025)
**Role**: Complete PyTorch Implementation
**Contribution**:
- Complete PyTorch backend implementation
- All 26 test problems in PyTorch
- 9 datasets, 9 architectures
- Comprehensive documentation (122 KB)
- 175+ unit tests
- Migration guides and examples

**Project Phases Completed**:
1. Foundation (base classes, configuration)
2. Simple datasets (MNIST, Fashion-MNIST, CIFAR)
3. Simple architectures (Logistic Regression, MLP, 2C2D)
4. Convolutional networks (3C3D, VGG, All-CNN-C)
5. Advanced architectures (Wide ResNet, Inception V3, VAE)
6. RNN and specialized problems (Character RNN, quadratic, 2D)
7. Complete test suite (175+ tests)
8. Comprehensive documentation
9. Final validation and release preparation

---

## Acknowledgments

### Framework Contributors

**PyTorch Team**
- For creating an excellent deep learning framework
- For comprehensive documentation and community support
- For maintaining torchvision model implementations

**TensorFlow Team**
- For the original framework that powered DeepOBS
- For TensorFlow 1.x which the original implementation used

### Research Community

**Optimizer Researchers**
- Users who provided feedback on DeepOBS
- Researchers who cited the DeepOBS paper
- Community members who reported bugs and suggested improvements

### Dataset Providers

- **MNIST**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **Fashion-MNIST**: Han Xiao, Kashif Rasul, Roland Vollgraf (Zalando Research)
- **CIFAR-10/100**: Alex Krizhevsky, Vinod Nair, Geoffrey Hinton
- **SVHN**: Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng
- **ImageNet**: ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
- **Tolstoi**: Project Gutenberg (War and Peace text)

### Architecture References

- **VGG**: Karen Simonyan and Andrew Zisserman
- **Inception**: Christian Szegedy et al.
- **ResNet**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Wide ResNet**: Sergey Zagoruyko and Nikos Komodakis
- **VAE**: Diederik P. Kingma and Max Welling

---

## Third-Party Libraries

DeepOBS relies on several excellent open-source libraries:

### Core Dependencies
- **NumPy**: Array computing and numerical operations
- **Pandas**: Data analysis and result management
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization

### Framework Dependencies
- **PyTorch**: Deep learning framework (PyTorch version)
- **torchvision**: PyTorch vision utilities and models
- **TensorFlow**: Deep learning framework (TensorFlow version)

### Development Tools
- **pytest**: Testing framework
- **pytest-cov**: Code coverage reporting
- **black**: Code formatting
- **flake8**: Code linting

---

## How to Contribute

We welcome contributions to DeepOBS! Here's how you can help:

### Reporting Bugs

1. Check existing issues: https://github.com/fsschneider/DeepOBS/issues
2. Create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs. actual behavior
   - Environment details (Python version, PyTorch version, OS)
   - Complete error traceback
   - Minimal reproducible example

### Suggesting Enhancements

1. Check existing feature requests
2. Open an issue describing:
   - The enhancement
   - Use cases
   - Expected benefits
   - Potential implementation approach

### Contributing Code

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Follow PEP 8 style guidelines
7. Add documentation for new features
8. Commit your changes (`git commit -m 'Add amazing feature'`)
9. Push to the branch (`git push origin feature/amazing-feature`)
10. Open a Pull Request

### Contributing Documentation

1. Documentation improvements are always welcome
2. Fix typos, clarify explanations, add examples
3. Follow the same process as code contributions
4. Ensure markdown formatting is correct

### Code Style Guidelines

- **PEP 8**: Follow Python PEP 8 style guide
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Add type hints where applicable
- **Comments**: Add comments for complex logic
- **Tests**: Write tests for new functionality
- **Documentation**: Update documentation for changes

### Testing Guidelines

- Write unit tests for new features
- Ensure all tests pass before submitting
- Add integration tests for complex features
- Maintain test coverage above 80%

---

## Citation

If you use DeepOBS in your research, please cite:

```bibtex
@inproceedings{schneider2019deepobs,
  title={DeepOBS: A Deep Learning Optimizer Benchmark Suite},
  author={Schneider, Frank and Balles, Lukas and Hennig, Philipp},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=rJg6ssC5Y7}
}
```

For the PyTorch implementation specifically:

```bibtex
@software{deepobs_pytorch_2025,
  title={DeepOBS PyTorch: Deep Learning Optimizer Benchmark Suite for PyTorch},
  author={Schneider, Frank and Bahde, Aaron and Contributors},
  year={2025},
  version={1.2.0},
  url={https://github.com/fsschneider/DeepOBS}
}
```

---

## License

DeepOBS is released under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2019 Frank Schneider, Lukas Balles, Philipp Hennig
Copyright (c) 2025 PyTorch Implementation Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

### General Inquiries
- **Email**: frank.schneider@tue.mpg.de
- **GitHub Issues**: https://github.com/fsschneider/DeepOBS/issues

### Bug Reports
- **GitHub Issues**: https://github.com/fsschneider/DeepOBS/issues

### Feature Requests
- **GitHub Issues**: https://github.com/fsschneider/DeepOBS/issues

### Research Collaboration
- **Email**: frank.schneider@tue.mpg.de

---

## Hall of Fame

Special recognition for significant contributions:

### 2019
- Frank Schneider, Lukas Balles, Philipp Hennig - Original DeepOBS publication

### 2020-2024
- Aaron Bahde - Led DeepOBS 1.2.0 development with improved interface

### 2025
- PyTorch Migration Team - Complete PyTorch implementation

---

## Thank You

Thank you to everyone who has contributed to DeepOBS, whether through code, documentation, bug reports, feature requests, or usage in research. Your contributions make DeepOBS better for the entire deep learning community.

Special thanks to the Max Planck Institute for Intelligent Systems for supporting the original DeepOBS development and research.

---

**Last Updated**: 2025-12-15
**Version**: 1.2.0 (PyTorch Support)
