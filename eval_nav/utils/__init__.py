# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Utility functions for navigation evaluation."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "load_policy_from_checkpoint",
    "check_success",
    "check_failure",
    "check_task_status",
    "StateLogger",
]

_LAZY = {
    "load_policy_from_checkpoint": (".policy_loader", "load_policy_from_checkpoint"),
    "check_success": (".task_checker", "check_success"),
    "check_failure": (".task_checker", "check_failure"),
    "check_task_status": (".task_checker", "check_task_status"),
    "StateLogger": (".state_logger", "StateLogger"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value
