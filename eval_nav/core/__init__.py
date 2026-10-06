# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Core evaluation engine for navigation environments.

The evaluator imports Isaac Lab. Names that need it are resolved on first use
so scorer and benchmark imports stay usable without a simulator.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "NavigationEvaluator",
    "EpisodeRunner",
    "EvaluationReporter",
    "V1Scorer",
    "V2Scorer",
    "V3Scorer",
    "V4Scorer",
    "get_scorer",
]

_LAZY = {
    "NavigationEvaluator": (".evaluator", "NavigationEvaluator"),
    "EpisodeRunner": (".episode_runner", "EpisodeRunner"),
    "EvaluationReporter": (".reporter", "EvaluationReporter"),
    "V1Scorer": (".scorer", "V1Scorer"),
    "V2Scorer": (".scorer", "V2Scorer"),
    "V3Scorer": (".scorer", "V3Scorer"),
    "V4Scorer": (".scorer", "V4Scorer"),
    "get_scorer": (".scorer", "get_scorer"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value
