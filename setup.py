# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Setup script for nav-eval package."""

from setuptools import find_packages, setup

setup(
    name="nav-eval",
    version="0.1.0",
    description="Navigation Evaluation Framework for IsaacLab",
    author="Nepher Team",
    license="BSD-3-Clause",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.20.0",
        "pyyaml>=6.0",
        "torch>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "nav-eval=nav_eval.scripts.evaluate:main",
        ],
    },
)

