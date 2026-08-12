# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""RunJump-specific event derivation from raw telemetry series.

All detector constants live here (sourced by HumanoidRunJumpScorer.to_dict).
eval_compat only supplies raw physics; this module reconstructs apex clearance,
landing impact, and post-crossing stability offline / online from the series.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

MODE_RUN = 0
MODE_JUMP = 1


@dataclass(frozen=True)
class RunJumpDerivationCfg:
    """Detector configuration for RunJump event derivation."""

    contact_force_n: float = 5.0
    impact_window_steps: int = 8
    stability_probe_steps: int = 50
    stable_min_height: float = 0.55
    stable_min_cos_tilt: float = 0.83
    stable_min_vx: float = 1.0
    clear_margin_m: float = 0.05

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


DEFAULT_CFG = RunJumpDerivationCfg()


def _both_feet_contact(forces: np.ndarray, threshold: float) -> np.ndarray:
    """forces [T,2,3] → bool [T] both feet above threshold."""
    norms = np.linalg.norm(forces, axis=-1)  # [T,2]
    return (norms[:, 0] > threshold) & (norms[:, 1] > threshold)


def derive_runjump_episode(
    series: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
    cfg: RunJumpDerivationCfg | None = None,
) -> dict[str, Any]:
    """Derive RunJump event aggregates from a raw per-step series.

    Returns unnormalized extras:
      apex_clearance_m (list), mean_apex_clearance_m
      landing_peak_vz (list), mean_landing_peak_vz
      crossings, stable_crossings, stable_clear_rate, final_cleared_count
    """
    cfg = cfg or DEFAULT_CFG
    meta = metadata or {}
    origin = np.asarray(meta.get("env_origin", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    obstacle_xs = [float(x) for x in meta.get("obstacle_xs", [])]
    obstacle_hs = [float(h) for h in meta.get("obstacle_hs", [])]
    obstacle_ts = [float(t) for t in meta.get("obstacle_ts", [0.20] * len(obstacle_xs))]
    if len(obstacle_ts) < len(obstacle_xs):
        obstacle_ts = obstacle_ts + [0.20] * (len(obstacle_xs) - len(obstacle_ts))
    num_obstacles = int(meta.get("num_obstacles", len(obstacle_xs)))

    out: dict[str, Any] = {
        "apex_clearance_m": [],
        "landing_peak_vz": [],
        "crossings": 0.0,
        "stable_crossings": 0.0,
        "stable_clear_rate": 0.0,
        "final_cleared_count": 0.0,
    }

    mode = series.get("hl_mode")
    pos_w = series.get("root_pos_w")
    lin_w = series.get("root_lin_vel_w")
    ankle = series.get("ankle_pos_w")
    forces = series.get("foot_contact_force_w")
    grav = series.get("projected_gravity_b")
    cleared = series.get("cleared_count")
    obs_idx = series.get("obstacle_index")

    t_len = 0
    for arr in (mode, pos_w, lin_w, ankle, forces):
        if arr is not None:
            t_len = max(t_len, len(arr))
    if t_len == 0:
        return out

    if mode is None:
        mode = np.zeros(t_len, dtype=np.int64)
    else:
        mode = np.asarray(mode, dtype=np.int64).reshape(-1)

    if pos_w is None:
        return out
    pos_w = np.asarray(pos_w, dtype=np.float64)
    s = pos_w[:, 0] - origin[0]
    root_z = pos_w[:, 2]
    vx_w = (
        np.asarray(lin_w, dtype=np.float64)[:, 0]
        if lin_w is not None
        else np.zeros(t_len, dtype=np.float64)
    )
    vz_w = (
        np.asarray(lin_w, dtype=np.float64)[:, 2]
        if lin_w is not None
        else np.zeros(t_len, dtype=np.float64)
    )

    if ankle is not None:
        ankle = np.asarray(ankle, dtype=np.float64)
        ankle_z_max = ankle[:, :, 2].max(axis=-1)
    else:
        ankle_z_max = root_z.copy()

    if forces is not None:
        forces = np.asarray(forces, dtype=np.float64)
        both = _both_feet_contact(forces, cfg.contact_force_n)
    else:
        both = np.zeros(t_len, dtype=bool)

    if grav is not None:
        grav = np.asarray(grav, dtype=np.float64)
        cos_tilt = -grav[:, 2]
    else:
        cos_tilt = np.ones(t_len, dtype=np.float64)

    # ------------------------------------------------------------------
    # Apex clearance: one sample per JUMP segment
    # ------------------------------------------------------------------
    apex_clearances: list[float] = []
    in_jump = False
    peak_z = 0.0
    latched_obs = 0
    for t in range(t_len):
        is_jump = int(mode[t]) == MODE_JUMP
        if is_jump and not in_jump:
            in_jump = True
            peak_z = float(ankle_z_max[t])
            if obs_idx is not None and t < len(obs_idx):
                latched_obs = int(obs_idx[t])
            else:
                latched_obs = _nearest_obstacle_index(float(s[t]), obstacle_xs)
        elif is_jump and in_jump:
            peak_z = max(peak_z, float(ankle_z_max[t]))
        elif (not is_jump) and in_jump:
            in_jump = False
            hurdle_h = 0.0
            if 0 <= latched_obs < len(obstacle_hs):
                hurdle_h = float(obstacle_hs[latched_obs])
            elif obstacle_hs:
                # Fallback: obstacle whose front face is nearest to apex s.
                k = _nearest_obstacle_index(float(s[t]), obstacle_xs)
                if 0 <= k < len(obstacle_hs):
                    hurdle_h = float(obstacle_hs[k])
            apex_clearances.append(float(peak_z - hurdle_h))
    if in_jump:
        # Episode ended mid-jump — still record the partial peak.
        hurdle_h = 0.0
        if 0 <= latched_obs < len(obstacle_hs):
            hurdle_h = float(obstacle_hs[latched_obs])
        apex_clearances.append(float(peak_z - hurdle_h))

    out["apex_clearance_m"] = apex_clearances
    if apex_clearances:
        out["mean_apex_clearance_m"] = float(np.mean(apex_clearances))

    # ------------------------------------------------------------------
    # Landing impact: dual-foot rising edge while JUMP → peak -vz window
    # ------------------------------------------------------------------
    landing_vz: list[float] = []
    prev_both = False
    impact_timer = 0
    impact_peak = 0.0
    for t in range(t_len):
        is_jump = int(mode[t]) == MODE_JUMP
        cur_both = bool(both[t])
        rising = cur_both and (not prev_both) and is_jump
        if rising:
            impact_timer = int(cfg.impact_window_steps)
            impact_peak = 0.0
        if impact_timer > 0:
            down = max(0.0, -float(vz_w[t]))
            impact_peak = max(impact_peak, down)
            impact_timer -= 1
            if impact_timer == 0:
                landing_vz.append(float(impact_peak))
        prev_both = cur_both

    out["landing_peak_vz"] = landing_vz
    if landing_vz:
        out["mean_landing_peak_vz"] = float(np.mean(landing_vz))

    # ------------------------------------------------------------------
    # Crossings + post-crossing stability probe
    # ------------------------------------------------------------------
    crossings = 0
    stable_crossings = 0
    crossed = [False] * num_obstacles
    stab_timer = 0
    pending_stable = False

    for t in range(t_len):
        if pending_stable and stab_timer > 0:
            stab_timer -= 1
            if stab_timer == 0:
                pending_stable = False
                upright = (
                    float(root_z[t]) > cfg.stable_min_height
                    and float(cos_tilt[t]) > cfg.stable_min_cos_tilt
                    and float(vx_w[t]) > cfg.stable_min_vx
                )
                if upright:
                    stable_crossings += 1

        for k in range(min(num_obstacles, len(obstacle_xs))):
            if crossed[k]:
                continue
            back = float(obstacle_xs[k]) + float(obstacle_ts[k]) + float(cfg.clear_margin_m)
            if float(s[t]) > back:
                crossed[k] = True
                crossings += 1
                pending_stable = True
                stab_timer = int(cfg.stability_probe_steps)

    out["crossings"] = float(crossings)
    out["stable_crossings"] = float(stable_crossings)
    if crossings > 0:
        out["stable_clear_rate"] = float(stable_crossings / crossings)
    else:
        out["stable_clear_rate"] = 0.0

    if cleared is not None and len(cleared) > 0:
        out["final_cleared_count"] = float(cleared[-1])
    else:
        out["final_cleared_count"] = float(crossings)

    return out


def _nearest_obstacle_index(s: float, xs: list[float]) -> int:
    if not xs:
        return -1
    # Prefer the obstacle whose front face is at or just behind s (being jumped).
    best = 0
    best_dist = abs(float(xs[0]) - s)
    for i, x in enumerate(xs):
        d = abs(float(x) - s)
        if d < best_dist:
            best = i
            best_dist = d
    return best
