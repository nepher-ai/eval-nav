# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Error handling and evaluation states for navigation evaluation."""

from enum import Enum
from typing import Any


class EvaluationStatus(str, Enum):
    """Evaluation outcome status."""
    
    SUCCESS = "SUCCESS"
    ENV_ERROR = "ENV_ERROR"
    EVAL_ERROR = "EVAL_ERROR"
    TIMEOUT = "TIMEOUT"


class EvaluationError(Exception):
    """Base exception for evaluation errors."""
    
    def __init__(self, message: str, status: EvaluationStatus, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.status = status
        self.details = details or {}


class EnvironmentError(EvaluationError):
    """Raised when environment cannot be loaded."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, EvaluationStatus.ENV_ERROR, details)


class EvaluationRuntimeError(EvaluationError):
    """Raised when evaluation fails at runtime."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, EvaluationStatus.EVAL_ERROR, details)


class TimeoutError(EvaluationError):
    """Raised when evaluation exceeds time limit."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, EvaluationStatus.TIMEOUT, details)

