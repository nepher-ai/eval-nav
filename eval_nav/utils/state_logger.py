# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State logging utilities for navigation evaluation.

This module provides functionality to log robot state data (position, yaw, waypoints, etc.)
during episode execution and save as .npy files per environment index.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class StateLogger:
    """Logs robot state data during episode execution."""
    
    def __init__(self, log_dir: Path | str | None = None, enabled: bool = True):
        """Initialize state logger.
        
        Args:
            log_dir: Directory to save log files. If None, logging is disabled.
            enabled: Whether logging is enabled.
        """
        self.enabled = enabled and (log_dir is not None)
        self.log_dir = Path(log_dir) if log_dir else None
        
        if self.enabled and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-environment state buffers (key: (episode_id, env_idx))
        # Stores data as dict of lists for efficient batching (each key maps to a list of values)
        self._state_buffers: dict[tuple[int, int], dict[str, list[Any]]] = {}
        # Per-environment metadata (logged once per episode) (key: (episode_id, env_idx))
        self._metadata_buffers: dict[tuple[int, int], dict[str, Any] | None] = {}
    
    def reset(self, episode_id: int, env_idx: int | None = None, env: Any = None) -> None:
        """Reset logger for a new episode.
        
        Args:
            episode_id: Episode identifier.
            env_idx: Environment index (for vectorized envs). If None, uses 0.
            env: Gymnasium environment (optional, used to extract metadata).
        """
        if not self.enabled:
            return
        
        if env_idx is None:
            env_idx = 0
        
        # Create unique key for this episode+env combination
        key = (episode_id, env_idx)
        self._state_buffers[key] = {}
        
        # Extract and store metadata once per episode (optimizes file size)
        if env is not None and hasattr(env, "_log_metadata"):
            metadata = env._log_metadata(env_idx=env_idx)
            self._metadata_buffers[key] = metadata if metadata else None
        else:
            self._metadata_buffers[key] = None
    
    def log_step(
        self,
        env: Any,
        episode_id: int,
        step: int,
        env_idx: int | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Log robot state for a single step.
        
        Args:
            env: Gymnasium environment.
            episode_id: Episode identifier.
            step: Current step number.
            env_idx: Environment index (for vectorized envs). If None, uses 0.
            info: Info dictionary from environment step.
        """
        if not self.enabled:
            return
        
        if env_idx is None:
            env_idx = 0
        
        key = (episode_id, env_idx)
        
        # Collect robot state - use standardized method if available
        if hasattr(env, "_log_state"):
            # Use centralized state extraction from EvalCompatEnv
            state_data = env._log_state(env_idx=env_idx, info=info)
        else:
            # Fallback for non-wrapped environments
            state_data = {}
        
        # Store step number (only variable per step)
        state_data["step"] = step
        # Note: episode_id and env_idx are NOT stored per step (they're in the key)

        # Initialize buffer if needed
        if key not in self._state_buffers:
            self._state_buffers[key] = {}
        
        # Append each field to its respective list for efficient batching
        for field_name, field_value in state_data.items():
            if field_name not in self._state_buffers[key]:
                self._state_buffers[key][field_name] = []
            
            # Convert tensors to numpy immediately to save memory
            if isinstance(field_value, torch.Tensor):
                self._state_buffers[key][field_name].append(field_value.cpu().numpy())
            elif isinstance(field_value, (list, np.ndarray)):
                self._state_buffers[key][field_name].append(np.asarray(field_value))
            else:
                self._state_buffers[key][field_name].append(field_value)
    
    def save(
        self,
        episode_id: int,
        scene: str | int,
        seed: int,
        env_idx: int | None = None,
        nav_env_id: str | None = None,
    ) -> Path | None:
        """Save logged state data to .npy file.
        
        Args:
            episode_id: Episode identifier.
            scene: Scene ID.
            seed: Random seed.
            env_idx: Environment index (for vectorized envs). If None, uses 0.
            
        Returns:
            Path to saved file, or None if not enabled.
        """
        if not self.enabled or self.log_dir is None:
            return None
        
        if env_idx is None:
            env_idx = 0
        
        key = (episode_id, env_idx)
        
        if key not in self._state_buffers or len(self._state_buffers[key]) == 0:
            logger.warning(f"No state data to save for episode {episode_id}, env_idx {env_idx}")
            return None
        
        # Convert batched lists to efficient numpy arrays
        state_dict = self._convert_to_numpy(self._state_buffers[key])
        
        # Get metadata for this episode/env (stored once per episode)
        metadata = self._metadata_buffers.get(key)

        # Create filename (short but informative)
        nav_env_part = f"{nav_env_id}_" if nav_env_id is not None else ""
        filename = f"ep{episode_id}_{nav_env_part}sc{scene}_sd{seed}_e{env_idx}.npy"
        filepath = self.log_dir / filename
        
        # Save as .npy file with metadata included
        # Structure: {'state': state_dict, 'metadata': metadata, 'episode_id': episode_id, 'env_idx': env_idx}
        # episode_id and env_idx stored once at the top level (not per step)
        save_data = {
            "state": state_dict,
            "metadata": metadata,
            "episode_id": episode_id,
            "env_idx": env_idx,
        }
        np.save(filepath, save_data)
        logger.debug(f"Saved state log to {filepath} (episode_id={episode_id}, env_idx={env_idx})")
        
        # Clear buffers for this episode/env
        del self._state_buffers[key]
        if key in self._metadata_buffers:
            del self._metadata_buffers[key]

        return filepath
    
    
    def _convert_to_numpy(self, state_buffer: dict[str, list[Any]]) -> dict[str, np.ndarray]:
        """Convert batched state lists to efficient numpy arrays.
        
        Args:
            state_buffer: Dictionary mapping field names to lists of values.
            
        Returns:
            Dictionary mapping field names to numpy arrays (batched efficiently).
        """
        if not state_buffer:
            return {}
        
        state_dict = {}
        num_steps = None
        
        # Determine number of steps from first field
        for field_name, field_list in state_buffer.items():
            if len(field_list) > 0:
                num_steps = len(field_list)
                break
        
        if num_steps is None or num_steps == 0:
            return {}
        
        # Convert each field to a numpy array efficiently
        for field_name, field_list in state_buffer.items():
            if len(field_list) != num_steps:
                logger.warning(f"Field '{field_name}' has {len(field_list)} entries, expected {num_steps}")
                continue
            
            # Stack arrays efficiently
            try:
                # Try to stack as numpy array (works for numeric arrays)
                stacked = np.stack(field_list) if field_list else np.array([])
                state_dict[field_name] = stacked
            except (ValueError, TypeError):
                # For scalar values or mixed types, create array of objects
                # This handles cases like floats, ints, bools, or arrays of different shapes
                state_dict[field_name] = np.array(field_list)
        
        return state_dict

