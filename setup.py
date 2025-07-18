#!/usr/bin/env python3
"""
Setup script for TSLies package
"""

from setuptools import setup, find_packages

setup(
    name="tslies",
    version="1.0.0",
    description="TSLies: Time Series Anomaly Detection Framework",
    author="Andrea Adelfio",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy", 
        "scikit-learn",
        "tensorflow",
        "tensorflow-probability",
        "keras",
        "matplotlib",
        "seaborn",
        "scipy",
        "tqdm",
    ],
    package_data={
        '': ['*.txt', '*.csv', '*.fits', '*.png', '*.md'],
    },
    include_package_data=True,
)
