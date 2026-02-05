# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Setup script for eval-nav package."""

from setuptools import find_packages, setup

setup(
    name="eval-nav",
    version="0.1.0",
    description="Navigation Evaluation Framework for IsaacLab",
    author="Nepher Team",
    license="Proprietary",
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
            "eval-nav=eval_nav.scripts.evaluate:main",
        ],
    },
)

