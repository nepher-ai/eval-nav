# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Navigation Evaluation Framework for IsaacLab.

A minimal but strong evaluation system for navigation environments with:
- Fixed evaluation campaigns
- Deterministic execution
- V1 scoring system
- Comprehensive metric collection
- Structured failure handling
"""

# Public API - main components
from .domain.config import EvalConfig
from .core.evaluator import NavigationEvaluator
from .core.reporter import EvaluationReporter

# Backward compatibility - expose submodules
from . import core, domain, managers, utils

__all__ = [
    # Public API
    "EvalConfig",
    "NavigationEvaluator",
    "EvaluationReporter",
    # Submodules for extensibility
    "core",
    "domain",
    "managers",
    "utils",
]

__version__ = "0.1.0"
