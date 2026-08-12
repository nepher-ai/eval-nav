# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Task-agnostic kinematic derivation from raw root state series."""

from __future__ import annotations

from typing import Any

import numpy as np


def derive_kinematics_episode(
    series: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduce raw root series into episode-level kinematic aggregates.

    Expects (optional keys are skipped when absent):
      root_lin_vel_b [T,3], root_ang_vel_b [T,3], root_pos_w [T,3]
      metadata.env_origin [3]
    """
    meta = metadata or {}
    origin = np.asarray(meta.get("env_origin", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)

    out: dict[str, Any] = {}

    lin_b = series.get("root_lin_vel_b")
    if lin_b is not None and len(lin_b) > 0:
        lin_b = np.asarray(lin_b, dtype=np.float64)
        speed_2d = np.linalg.norm(lin_b[:, :2], axis=-1)
        lat = np.abs(lin_b[:, 1])
        vert = np.abs(lin_b[:, 2])
        out["mean_speed"] = float(speed_2d.mean())
        out["max_speed"] = float(speed_2d.max())
        out["speed_std"] = float(speed_2d.std())
        out["mean_lateral_speed"] = float(lat.mean())
        out["max_lateral_speed"] = float(lat.max())
        out["mean_vertical_speed"] = float(vert.mean())

    ang_b = series.get("root_ang_vel_b")
    if ang_b is not None and len(ang_b) > 0:
        ang_b = np.asarray(ang_b, dtype=np.float64)
        yaw = np.abs(ang_b[:, 2])
        rp = np.linalg.norm(ang_b[:, :2], axis=-1)
        out["mean_yaw_rate"] = float(yaw.mean())
        out["max_yaw_rate"] = float(yaw.max())
        out["mean_angular_speed"] = float(yaw.mean())
        out["angular_speed_std"] = float(yaw.std())
        out["mean_roll_pitch_rate"] = float(rp.mean())

    pos_w = series.get("root_pos_w")
    if pos_w is not None and len(pos_w) > 0:
        pos_w = np.asarray(pos_w, dtype=np.float64)
        s = pos_w[:, 0] - origin[0]
        y = pos_w[:, 1] - origin[1]
        abs_y = np.abs(y)

        s_max = np.maximum.accumulate(s)
        # Progress delta: advance of the monotone potential (matches HL env).
        progress = np.empty_like(s_max)
        progress[0] = max(float(s_max[0]), 0.0)
        if len(s_max) > 1:
            progress[1:] = np.maximum(s_max[1:] - s_max[:-1], 0.0)

        out["mean_abs_lateral_offset_m"] = float(abs_y.mean())
        out["max_abs_lateral_offset_m"] = float(abs_y.max())
        y2 = abs_y * abs_y
        w_sum = float(progress.sum())
        if w_sum > 1e-8:
            out["rms_lateral_offset_m"] = float(np.sqrt(np.sum(progress * y2) / w_sum))
        else:
            out["rms_lateral_offset_m"] = float(np.sqrt(y2.mean()))

    hl = series.get("hl_action")
    if hl is not None and len(hl) > 0:
        hl = np.asarray(hl, dtype=np.float64)
        if hl.ndim == 1:
            norms = np.abs(hl)
        else:
            norms = np.linalg.norm(hl, axis=-1)
        out["mean_action_l2"] = float(norms.mean())

    return out
