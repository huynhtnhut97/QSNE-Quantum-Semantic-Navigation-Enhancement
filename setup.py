"""Setup script for the qsne package."""

from setuptools import find_packages, setup

setup(
    name="qsne",
    version="0.1.0",
    description=(
        "Quantum-Semantic Navigation Enhancement (QSNE): a learning framework "
        "for robust ground-robot navigation under partial observability."
    ),
    author="Truong Nhut Huynh, Caiden Sivak, Hector Gutierrez, Kim-Doang Nguyen",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(exclude=("scripts", "scripts.*")),
    install_requires=[
        "numpy>=1.24",
        "torch>=1.13",
        "pennylane>=0.32",
    ],
    extras_require={
        "llm": ["openai>=1.6", "transformers>=4.36"],
        "ros": [],  # ROS Noetic must be installed at the system level.
    },
)
