# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success checking utilities for navigation evaluation.

This module provides task-aware success checking that can be extended
for different navigation tasks (waypoint navigation, goal navigation, etc.)
and different robots.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import torch


def check_success(
    env: gym.Env,
    info: dict[str, Any],
    task_name: str,
    env_idx: int | None = None,
    current_success: bool = False,
) -> bool:
    """Check if an episode succeeded based on task-specific success criteria.
    
    This function implements task-aware success checking that can vary across
    different navigation tasks (waypoint navigation, goal navigation, etc.).
    
    Args:
        env: The gymnasium environment.
        info: Info dictionary from the environment step.
        task_name: Task name (e.g., "Nepher-Animal-WaypointNav-Envs-Play-v0").
        env_idx: Environment index for vectorized environments (None for single env).
        current_success: Current success status (will be OR'd with new check).
        
    Returns:
        Boolean indicating success status.
    """
    success = current_success
    
    # Determine task type from task name
    task_name_lower = task_name.lower()
    is_waypoint_nav = "waypoint" in task_name_lower
    is_goal_nav = "goal" in task_name_lower or "target" in task_name_lower
    
    # Method 1: Check info dict for success flag (universal)
    if "success" in info or "successes" in info:
        success_val = info.get("success", False) or info.get("successes", False)
        if torch.is_tensor(success_val):
            if env_idx is not None:
                success = bool(success_val[env_idx].item()) or success
            else:
                success = bool(success_val.item()) or success
        else:
            success = bool(success_val) or success
    
    # Method 2: Task-specific success checks
    if is_waypoint_nav:
        success = _check_waypoint_success(env, info, env_idx, success)
    elif is_goal_nav:
        success = _check_goal_success(env, info, env_idx, success)
    
    # Method 3: Fallback - check for any task-specific success indicators
    # This can be extended for other task types
    if not success:
        success = _check_fallback_success(info, env_idx, success)
    
    return success


def _check_waypoint_success(
    env: gym.Env,
    info: dict[str, Any],
    env_idx: int | None,
    current_success: bool,
) -> bool:
    """Check success for waypoint navigation tasks.
    
    Args:
        env: The gymnasium environment.
        info: Info dictionary from the environment step.
        env_idx: Environment index for vectorized environments.
        current_success: Current success status.
        
    Returns:
        Updated success status.
    """
    success = current_success
    
    # Waypoint navigation: check if all waypoints are reached
    if hasattr(env.unwrapped, "command_manager"):
        try:
            waypoint_term = env.unwrapped.command_manager.get_term("waypoints")
            if hasattr(waypoint_term, "all_waypoints_reached"):
                all_reached = waypoint_term.all_waypoints_reached
                if torch.is_tensor(all_reached):
                    if env_idx is not None:
                        success = bool(all_reached[env_idx].item()) or success
                    else:
                        success = bool(all_reached.item()) or success
                else:
                    success = bool(all_reached) or success
        except Exception as e:
            logging.debug(f"Failed to check waypoint success: {e}")
    
    return success


def _check_goal_success(
    env: gym.Env,
    info: dict[str, Any],
    env_idx: int | None,
    current_success: bool,
) -> bool:
    """Check success for goal navigation tasks.
    
    Args:
        env: The gymnasium environment.
        info: Info dictionary from the environment step.
        env_idx: Environment index for vectorized environments.
        current_success: Current success status.
        
    Returns:
        Updated success status.
    """
    success = current_success
    
    # Goal navigation: check goal reached status
    if hasattr(env.unwrapped, "command_manager"):
        try:
            # Try to get goal command term
            goal_term = env.unwrapped.command_manager.get_term("goal")
            if hasattr(goal_term, "goal_reached"):
                goal_reached = goal_term.goal_reached
                if torch.is_tensor(goal_reached):
                    if env_idx is not None:
                        success = bool(goal_reached[env_idx].item()) or success
                    else:
                        success = bool(goal_reached.item()) or success
                else:
                    success = bool(goal_reached) or success
        except Exception as e:
            logging.debug(f"Failed to check goal success: {e}")
    
    return success


def _check_fallback_success(
    info: dict[str, Any],
    env_idx: int | None,
    current_success: bool,
) -> bool:
    """Check for fallback success indicators in info dict.
    
    Args:
        info: Info dictionary from the environment step.
        env_idx: Environment index for vectorized environments.
        current_success: Current success status.
        
    Returns:
        Updated success status.
    """
    success = current_success
    
    # Check for other common success indicators in info
    for key in ["task_success", "episode_success", "completed"]:
        if key in info:
            success_val = info[key]
            if torch.is_tensor(success_val):
                if env_idx is not None:
                    success = bool(success_val[env_idx].item()) or success
                else:
                    success = bool(success_val.item()) or success
            else:
                success = bool(success_val) or success
            if success:
                break
    
    return success

