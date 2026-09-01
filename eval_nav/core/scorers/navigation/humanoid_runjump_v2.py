# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Humanoid RunJump course scorer (v2).

Built on v1 helpers (clearance / landing / track maps) but replaces the nested
weight tree and fixed-budget time term with a flat performance sum led by
capped average course speed, then multiplies by a naturalness gate so
non-human motion (X-legs, extreme tilt, thrashing) drives an episode to ~0.

Formula
-------
    score = success_rate × mean over successful episodes of (N × P)

    P = 0.40×speed + 0.20×clearance + 0.15×land_stable
      + 0.15×land_impact + 0.10×track

    speed = ramp(course_length_m / elapsed_s, 0, 3.0)

    N = n_cross × n_posture × n_spin   (exactly 1.0 for human-like motion)

Inherited ``BASE``, ``W_TIME``, ``W_CLEAR_LAND``, ``W_SAFETY_ENERGY``,
``W_BODY_STAB``, ``W_ENERGY``, ``ACTION_L2_REF``, ``MAX_VERTICAL_SPEED``,
and ``MAX_ROLL_PITCH_RATE`` are unused in v2 and absent from ``to_dict``.

Used by
-------
    ``task_type: "navigation.humanoid.runjump"``, ``scoring_version: "v2"``
    → eval-nav/configs/task-humanoid-run-jump.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...telemetry.derive.naturalness import DEFAULT_CFG as NATURALNESS_DEFAULT_CFG
from ...telemetry.derive.naturalness import NaturalnessDerivationCfg
from .humanoid_runjump import HumanoidRunJumpScorer


class HumanoidRunJumpScorerV2(HumanoidRunJumpScorer):
    """Humanoid RunJump course scorer (v2): SR × mean(N × flat performance)."""

    VERSION: str = "v2"

    # Flat performance weights (sum to 1).
    W_SPEED: float = 0.40
    W_CLEARANCE_FLAT: float = 0.20
    W_LAND_STABLE_FLAT: float = 0.15
    W_LAND_IMPACT_FLAT: float = 0.15
    W_TRACK_FLAT: float = 0.10

    V_REF_MPS: float = 3.0

    # Naturalness deadbands: ramp(v, ok, bad).
    CROSS_DEPTH_OK_M: float = 0.05
    CROSS_DEPTH_BAD_M: float = 0.30
    CROSS_TIME_OK: float = 0.02
    CROSS_TIME_BAD: float = 0.20

    LATERAL_TILT_OK_DEG: float = 25.0
    LATERAL_TILT_BAD_DEG: float = 50.0
    TILT_OK_DEG: float = 60.0
    TILT_BAD_DEG: float = 90.0

    ROLL_PITCH_RATE_OK: float = 1.5
    ROLL_PITCH_RATE_BAD: float = 4.0
    YAW_RATE_OK: float = 1.0
    YAW_RATE_BAD: float = 3.0

    NATURALNESS_CFG: NaturalnessDerivationCfg = NATURALNESS_DEFAULT_CFG

    def compute_score(
        self,
        metrics: Any,
        max_episode_steps: int,
        episodes: list[Any] | None = None,
        *,
        max_episode_time_s: float | None = None,
    ) -> float:
        """Compute the RunJump v2 score ∈ [0, 1]."""
        if not episodes or metrics.successful_episodes == 0:
            return 0.0

        qualities = [
            self._episode_quality(ep, max_episode_steps, max_episode_time_s=max_episode_time_s)
            for ep in episodes
            if ep.success
        ]
        if not qualities:
            return 0.0
        return float(metrics.success_rate * float(np.mean(qualities)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "navigation.humanoid.runjump",
            "scoring_version": self.VERSION,
            "weights": {
                "speed": self.W_SPEED,
                "clearance": self.W_CLEARANCE_FLAT,
                "land_stable": self.W_LAND_STABLE_FLAT,
                "land_impact": self.W_LAND_IMPACT_FLAT,
                "track": self.W_TRACK_FLAT,
            },
            "thresholds": {
                "v_ref_mps": self.V_REF_MPS,
                "y_ref_m": self.Y_REF_M,
                "clearance_margin_m": self.CLEARANCE_MARGIN_M,
                "landing_vz_ref": self.LANDING_VZ_REF,
                "cross_depth_ok_m": self.CROSS_DEPTH_OK_M,
                "cross_depth_bad_m": self.CROSS_DEPTH_BAD_M,
                "cross_time_ok": self.CROSS_TIME_OK,
                "cross_time_bad": self.CROSS_TIME_BAD,
                "lateral_tilt_ok_deg": self.LATERAL_TILT_OK_DEG,
                "lateral_tilt_bad_deg": self.LATERAL_TILT_BAD_DEG,
                "tilt_ok_deg": self.TILT_OK_DEG,
                "tilt_bad_deg": self.TILT_BAD_DEG,
                "roll_pitch_rate_ok": self.ROLL_PITCH_RATE_OK,
                "roll_pitch_rate_bad": self.ROLL_PITCH_RATE_BAD,
                "yaw_rate_ok": self.YAW_RATE_OK,
                "yaw_rate_bad": self.YAW_RATE_BAD,
            },
            "derivation": self.DERIVATION_CFG.to_dict(),
            "naturalness_derivation": self.NATURALNESS_CFG.to_dict(),
        }

    def _episode_quality(
        self,
        ep: Any,
        max_episode_steps: int,
        *,
        max_episode_time_s: float | None,
    ) -> float:
        performance = self._performance(ep, max_episode_steps, max_episode_time_s)
        naturalness = self._naturalness(ep)
        return float(naturalness * performance)

    def _performance(
        self,
        ep: Any,
        max_episode_steps: int,
        max_episode_time_s: float | None,
    ) -> float:
        ex = ep.extra or {}
        speed = self._speed_score(ep, max_episode_steps, max_episode_time_s)
        clearance = self._clearance_score(ex)
        land_stable = self._clip01(ex.get("stable_clear_rate"))
        land_impact = self._landing_impact_score(ex)
        if land_impact is None:
            cleared = ex.get("final_cleared_count")
            if cleared is None and "cleared_count" in ex:
                cleared = ex.get("cleared_count")
            land_impact = 0.0 if (cleared is not None and float(cleared) > 0) else 0.0
        track = self._track(ep)
        return float(
            self.W_SPEED * speed
            + self.W_CLEARANCE_FLAT * clearance
            + self.W_LAND_STABLE_FLAT * land_stable
            + self.W_LAND_IMPACT_FLAT * float(land_impact)
            + self.W_TRACK_FLAT * track
        )

    def _course_speed(
        self,
        ep: Any,
        max_episode_steps: int,
        max_episode_time_s: float | None,
    ) -> float | None:
        """Average course speed in m/s, or None if length/time unavailable."""
        ex = ep.extra or {}
        length = ex.get("course_length_m")
        if length is None:
            length = ex.get("progress_s_m")
        if length is None or float(length) <= 0.0:
            return None

        elapsed: float | None = None
        if ep.completion_time is not None and float(ep.completion_time) > 0.0:
            elapsed = float(ep.completion_time)
        else:
            step_dt = ex.get("step_dt")
            if step_dt is not None and float(step_dt) > 0.0 and ep.steps > 0:
                elapsed = float(ep.steps) * float(step_dt)
            elif (
                max_episode_time_s
                and max_episode_time_s > 0
                and max_episode_steps > 0
                and ep.steps > 0
            ):
                elapsed = float(ep.steps) * (max_episode_time_s / max_episode_steps)
        if elapsed is None or elapsed <= 0.0:
            return None
        return float(length) / elapsed

    def _speed_score(
        self,
        ep: Any,
        max_episode_steps: int,
        max_episode_time_s: float | None,
    ) -> float:
        speed = self._course_speed(ep, max_episode_steps, max_episode_time_s)
        if speed is None:
            return 0.0
        return self._ramp(speed, 0.0, self.V_REF_MPS)

    def _naturalness(self, ep: Any) -> float:
        """Product of three deadband factors; fail-closed on missing telemetry."""
        ex = ep.extra or {}
        required = (
            "leg_cross_depth_m",
            "leg_cross_time_frac",
            "lateral_tilt_deg",
            "tilt_deg",
            "mean_roll_pitch_rate",
            "mean_yaw_rate",
        )
        if any(k not in ex for k in required):
            return 0.0
        n_cross = self._n_cross(ex)
        n_posture = self._n_posture(ex)
        n_spin = self._n_spin(ex)
        return float(n_cross * n_posture * n_spin)

    def _n_cross(self, ex: dict[str, Any]) -> float:
        depth = self._ramp(
            float(ex["leg_cross_depth_m"]),
            self.CROSS_DEPTH_OK_M,
            self.CROSS_DEPTH_BAD_M,
        )
        time_frac = self._ramp(
            float(ex["leg_cross_time_frac"]),
            self.CROSS_TIME_OK,
            self.CROSS_TIME_BAD,
        )
        return float(1.0 - max(depth, time_frac))

    def _n_posture(self, ex: dict[str, Any]) -> float:
        lateral = self._ramp(
            float(ex["lateral_tilt_deg"]),
            self.LATERAL_TILT_OK_DEG,
            self.LATERAL_TILT_BAD_DEG,
        )
        tilt = self._ramp(
            float(ex["tilt_deg"]),
            self.TILT_OK_DEG,
            self.TILT_BAD_DEG,
        )
        return float(1.0 - max(lateral, tilt))

    def _n_spin(self, ex: dict[str, Any]) -> float:
        rp = self._ramp(
            float(ex["mean_roll_pitch_rate"]),
            self.ROLL_PITCH_RATE_OK,
            self.ROLL_PITCH_RATE_BAD,
        )
        yaw = self._ramp(
            float(ex["mean_yaw_rate"]),
            self.YAW_RATE_OK,
            self.YAW_RATE_BAD,
        )
        return float(1.0 - max(rp, yaw))

    @staticmethod
    def _ramp(value: float, ok: float, bad: float) -> float:
        """clip01((value − ok) / (bad − ok)); 0 below ok, 1 at/above bad."""
        if bad <= ok:
            return 1.0 if value >= bad else 0.0
        return float(max(0.0, min(1.0, (float(value) - float(ok)) / (float(bad) - float(ok)))))
