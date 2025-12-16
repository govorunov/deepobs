# -*- coding: utf-8 -*-
"""Setup for the DeepOBS package"""

import setuptools


def readme():
    with open("README.md") as f:
        return f.read()


setuptools.setup(
    name="deepobs",
    version="1.2.0",
    description="Deep Learning Optimizer Benchmark Suite",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Frank Schneider, Lukas Balles and Philipp Hennig,",
    author_email="frank.schneider@tue.mpg.de",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "argparse",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "pytorch": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
        "all": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
    },
    scripts=[
        "deepobs/scripts/deepobs_prepare_data.sh",
        "deepobs/scripts/deepobs_get_baselines.sh",
        "deepobs/scripts/deepobs_plot_results.py",
    ],
    zip_safe=False,
    python_requires=">=3.6",
)
