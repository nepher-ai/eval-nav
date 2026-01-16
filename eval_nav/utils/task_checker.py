# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task status checking utilities for navigation evaluation.

This module provides task-aware status checking (completion and failure) that
uses the eval_compat wrapper to access environment state in a consistent way.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import torch


def check_task_status(
    env: gym.Env,
    info: dict[str, Any] | None = None,
    env_idx: int | None = None,
    current_success: bool = False,
    current_failure: bool = False,
) -> tuple[bool, bool]:
    """Check task completion and failure status using eval_compat wrapper.
    
    This function checks both task completion (success) and task failure
    using the eval_compat wrapper's properties.
    
    Args:
        env: The gymnasium environment (may be wrapped with EvalCompatEnv).
        info: Info dictionary from the environment step (optional).
        env_idx: Environment index for vectorized environments (None for single env).
        current_success: Current success status (will be OR'd with new check).
        current_failure: Current failure status (will be OR'd with new check).
        
    Returns:
        Tuple of (success, failure) booleans indicating task status.
    """
    success = current_success
    failure = current_failure
    
    # Use eval_compat wrapper
    task_completed = getattr(env, "task_completed", None)
    task_failed = getattr(env, "task_failed", None)
    
    if task_completed is not None or task_failed is not None:
        try:
            # Check task completion
            if task_completed is not None:
                if torch.is_tensor(task_completed):
                    if env_idx is not None:
                        success = bool(task_completed[env_idx].item()) or success
                    else:
                        if task_completed.numel() == 1:
                            success = bool(task_completed.item()) or success
                        else:
                            success = bool(task_completed[0].item()) or success
                else:
                    success = bool(task_completed) or success
            
            if task_failed is not None:
                if torch.is_tensor(task_failed):
                    if env_idx is not None:
                        failure = bool(task_failed[env_idx].item()) or failure
                    else:
                        if task_failed.numel() == 1:
                            failure = bool(task_failed.item()) or failure
                        else:
                            failure = bool(task_failed[0].item()) or failure
                else:
                    failure = bool(task_failed) or failure
        except Exception as e:
            logging.debug(f"Failed to check task status via eval_compat: {e}")
    else:
        raise ValueError("Environment does not have task completion or failure properties")
    
    return success, failure


def check_success(
    env: gym.Env,
    info: dict[str, Any] | None = None,
    task_name: str | None = None,
    env_idx: int | None = None,
    current_success: bool = False,
) -> bool:
    """Check if an episode succeeded (backward compatibility wrapper).
    
    This function maintains backward compatibility with the old check_success API
    while using the new task status checker internally.
    
    Args:
        env: The gymnasium environment.
        info: Info dictionary from the environment step.
        task_name: Task name (deprecated, kept for compatibility).
        env_idx: Environment index for vectorized environments (None for single env).
        current_success: Current success status (will be OR'd with new check).
        
    Returns:
        Boolean indicating success status.
    """
    success, _ = check_task_status(
        env=env,
        info=info,
        env_idx=env_idx,
        current_success=current_success,
        current_failure=False,
    )
    return success


def check_failure(
    env: gym.Env,
    info: dict[str, Any] | None = None,
    env_idx: int | None = None,
    current_failure: bool = False,
) -> bool:
    """Check if an episode failed.
    
    Args:
        env: The gymnasium environment.
        info: Info dictionary from the environment step.
        env_idx: Environment index for vectorized environments (None for single env).
        current_failure: Current failure status (will be OR'd with new check).
        
    Returns:
        Boolean indicating failure status.
    """
    _, failure = check_task_status(
        env=env,
        info=info,
        env_idx=env_idx,
        current_success=False,
        current_failure=current_failure,
    )
    return failure

