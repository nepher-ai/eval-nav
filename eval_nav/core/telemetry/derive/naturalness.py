# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Naturalness derivation from raw telemetry (leg crossing, torso tilt).

All detector constants live here (sourced by HumanoidRunJumpScorerV2.to_dict).
eval_compat only supplies raw physics; this module reconstructs unnormalized
aggregates. [0, 1] maps stay in the scorer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

MODE_RUN = 0
MODE_JUMP = 1


@dataclass(frozen=True)
class NaturalnessDerivationCfg:
    """Detector configuration for naturalness event derivation."""

    sustain_steps: int = 5

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


DEFAULT_CFG = NaturalnessDerivationCfg()


def _sustained_max(values: np.ndarray, window: int) -> float:
    """Max over t of min(values[t : t+window]) — ignores single-frame spikes."""
    if values.size == 0:
        return 0.0
    k = max(1, min(int(window), int(values.size)))
    if k == 1:
        return float(np.max(values))
    best = float("-inf")
    for t in range(int(values.size) - k + 1):
        best = max(best, float(np.min(values[t : t + k])))
    return float(best) if best > float("-inf") else 0.0


def _yaw_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Extract yaw (rad) from wxyz quaternions [T, 4]."""
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def derive_naturalness_episode(
    series: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
    cfg: NaturalnessDerivationCfg | None = None,
) -> dict[str, Any]:
    """Derive naturalness aggregates from a raw per-step series.

    Returns unnormalized extras:
      leg_cross_depth_m, leg_cross_time_frac, mean_ankle_sep_m, min_ankle_sep_m
      jump_leg_cross_depth_m (list), mean_jump_leg_cross_depth_m
      lateral_tilt_deg, tilt_deg, mean_lateral_tilt_deg
    """
    cfg = cfg or DEFAULT_CFG
    _ = metadata  # reserved for future course-frame helpers
    out: dict[str, Any] = {}

    ankle = series.get("ankle_pos_w")
    root_pos = series.get("root_pos_w")
    root_quat = series.get("root_quat_w")
    grav = series.get("projected_gravity_b")
    mode = series.get("hl_mode")

    # ------------------------------------------------------------------
    # Leg crossing via yaw-projected ankle separation
    # ------------------------------------------------------------------
    if ankle is not None and root_pos is not None and root_quat is not None:
        ankle = np.asarray(ankle, dtype=np.float64)
        root_pos = np.asarray(root_pos, dtype=np.float64)
        root_quat = np.asarray(root_quat, dtype=np.float64)
        t_len = min(len(ankle), len(root_pos), len(root_quat))
        if t_len > 0 and ankle.ndim == 3 and ankle.shape[1] >= 2:
            ankle = ankle[:t_len]
            root_pos = root_pos[:t_len]
            root_quat = root_quat[:t_len]

            yaw = _yaw_from_quat_wxyz(root_quat)
            # Body +y expressed in world (leftward when yaw=0).
            lat = np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)
            # Left minus right, projected on body lateral axis. >0 = uncrossed.
            delta_xy = ankle[:, 0, :2] - ankle[:, 1, :2]
            sep = np.sum(delta_xy * lat, axis=-1)
            depth = np.maximum(0.0, -sep)

            out["leg_cross_depth_m"] = _sustained_max(depth, cfg.sustain_steps)
            out["leg_cross_time_frac"] = float(np.mean(depth > 0.0))
            out["mean_ankle_sep_m"] = float(sep.mean())
            out["min_ankle_sep_m"] = float(sep.min())

            # Per-JUMP-segment peak crossing depth (parallel to apex samples).
            jump_depths: list[float] = []
            if mode is not None:
                mode_arr = np.asarray(mode, dtype=np.int64).reshape(-1)[:t_len]
                in_jump = False
                peak = 0.0
                for t in range(t_len):
                    is_jump = int(mode_arr[t]) == MODE_JUMP
                    if is_jump and not in_jump:
                        in_jump = True
                        peak = float(depth[t])
                    elif is_jump and in_jump:
                        peak = max(peak, float(depth[t]))
                    elif (not is_jump) and in_jump:
                        in_jump = False
                        jump_depths.append(peak)
                if in_jump:
                    jump_depths.append(peak)
            out["jump_leg_cross_depth_m"] = jump_depths
            if jump_depths:
                out["mean_jump_leg_cross_depth_m"] = float(np.mean(jump_depths))

    # ------------------------------------------------------------------
    # Torso tilt from projected gravity in body frame
    # ------------------------------------------------------------------
    if grav is not None:
        grav = np.asarray(grav, dtype=np.float64)
        if grav.ndim == 2 and grav.shape[0] > 0 and grav.shape[1] >= 3:
            gz = np.clip(-grav[:, 2], -1.0, 1.0)
            gy = np.clip(np.abs(grav[:, 1]), 0.0, 1.0)
            tilt_rad = np.arccos(gz)
            lat_tilt_rad = np.arcsin(gy)
            tilt_deg = np.degrees(tilt_rad)
            lat_tilt_deg = np.degrees(lat_tilt_rad)

            out["tilt_deg"] = _sustained_max(tilt_deg, cfg.sustain_steps)
            out["lateral_tilt_deg"] = _sustained_max(lat_tilt_deg, cfg.sustain_steps)
            out["mean_lateral_tilt_deg"] = float(lat_tilt_deg.mean())

    return out
