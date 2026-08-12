# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Humanoid RunJump course scorer (v1).

Evaluates a G1 high-level policy on the EnvHub obstacle-course benchmark.
Success-rate-amplified score that blends completion time with clearance /
landing quality, centerline tracking, and body safety / command energy
(no AMP discriminator).

All [0,1] maps and detector constants live here. ``eval_compat`` only emits
raw physics; ``eval_nav.core.telemetry`` derives unnormalized aggregates.

Formula
-------
    score = success_rate × (BASE + (1 − BASE) × mean_quality)

    BASE = 0.25

    quality (per successful episode):
        = 0.35 × time_eff + 0.30 × clear_land + 0.20 × track + 0.15 × safety_energy

    time_eff:
        Uses physical time when max_episode_time_s is provided; else steps.

            time_eff = max(0, 1 − completion_time / max_episode_time_s)

    clear_land:
        = 0.40 × clearance + 0.30 × land_stable + 0.30 × land_impact

        clearance   ← mean(clip01(apex_clearance_m / CLEARANCE_MARGIN_M))
                      or clip01(mean_apex_clearance_m / CLEARANCE_MARGIN_M)
        land_stable ← extra["stable_clear_rate"]
        land_impact ← mean(clip01(1 − landing_peak_vz / LANDING_VZ_REF))
        Missing any of these → that sub-term is 0.0.

    track:
        Progress-weighted RMS lateral offset from the +x centerline:

            track = clip01(1 − (rms_lateral_offset_m / Y_REF)²)
            Y_REF = 0.75 m

        Missing ``rms_lateral_offset_m`` → 0.0 (fail-closed).

    safety_energy:
        = 0.60 × body_stab + 0.40 × energy

        body_stab from mean_vertical_speed / mean_roll_pitch_rate
        energy from mean_action_l2
        Missing loco / action telemetry → 0.0.

Used by
-------
    ``task_type: "navigation.humanoid.runjump"``, ``scoring_version: "v1"``
    → eval-nav/configs/task-humanoid-run-jump.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ...telemetry.derive.runjump import DEFAULT_CFG, RunJumpDerivationCfg
from ..base import BaseScorer


class HumanoidRunJumpScorer(BaseScorer):
    """Humanoid RunJump course scorer (v1): SR × (time + clear_land + track + safety)."""

    VERSION: str = "v1"
    BASE: float = 0.25

    W_TIME: float = 0.35
    W_CLEAR_LAND: float = 0.30
    W_TRACK: float = 0.20
    W_SAFETY_ENERGY: float = 0.15

    W_CLEARANCE: float = 0.40
    W_LAND_STABLE: float = 0.30
    W_LAND_IMPACT: float = 0.30

    W_BODY_STAB: float = 0.60
    W_ENERGY: float = 0.40

    MAX_VERTICAL_SPEED: float = 1.5
    MAX_ROLL_PITCH_RATE: float = 2.0
    ACTION_L2_REF: float = 1.5
    Y_REF_M: float = 0.75

    # Migrated from eval_compat / episode_runner — scorer owns all maps.
    CLEARANCE_MARGIN_M: float = 0.15
    LANDING_VZ_REF: float = 3.5

    DERIVATION_CFG: RunJumpDerivationCfg = DEFAULT_CFG

    def __init__(self, max_normalized_time: float = 1.0) -> None:
        self.max_normalized_time = max_normalized_time

    def compute_score(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
        *,
        max_episode_time_s: float | None = None,
    ) -> float:
        """Compute the RunJump score ∈ [0, 1]."""
        if not episodes or metrics.successful_episodes == 0:
            return 0.0

        success_rate = metrics.success_rate
        mean_quality = self._mean_quality(
            episodes, max_episode_steps, max_episode_time_s=max_episode_time_s
        )
        return float(success_rate * (self.BASE + (1.0 - self.BASE) * mean_quality))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "navigation.humanoid.runjump",
            "scoring_version": self.VERSION,
            "weights": {
                "time_efficiency": self.W_TIME,
                "clear_land": self.W_CLEAR_LAND,
                "track": self.W_TRACK,
                "safety_energy": self.W_SAFETY_ENERGY,
                "clearance": self.W_CLEARANCE,
                "land_stable": self.W_LAND_STABLE,
                "land_impact": self.W_LAND_IMPACT,
                "body_stability": self.W_BODY_STAB,
                "energy": self.W_ENERGY,
            },
            "thresholds": {
                "max_vertical_speed": self.MAX_VERTICAL_SPEED,
                "max_roll_pitch_rate": self.MAX_ROLL_PITCH_RATE,
                "action_l2_ref": self.ACTION_L2_REF,
                "y_ref_m": self.Y_REF_M,
                "clearance_margin_m": self.CLEARANCE_MARGIN_M,
                "landing_vz_ref": self.LANDING_VZ_REF,
            },
            "derivation": self.DERIVATION_CFG.to_dict(),
            "base": self.BASE,
        }

    def _mean_quality(
        self,
        episodes: list[EpisodeMetrics],
        max_episode_steps: int,
        *,
        max_episode_time_s: float | None,
    ) -> float:
        qualities = [
            self._episode_quality(ep, max_episode_steps, max_episode_time_s=max_episode_time_s)
            for ep in episodes
            if ep.success
        ]
        return float(np.mean(qualities)) if qualities else 0.0

    def _episode_quality(
        self,
        ep: EpisodeMetrics,
        max_episode_steps: int,
        *,
        max_episode_time_s: float | None,
    ) -> float:
        time_eff = self._time_efficiency(ep, max_episode_steps, max_episode_time_s)
        clear_land = self._clear_land(ep)
        track = self._track(ep)
        safety_energy = self._safety_energy(ep)
        return float(
            self.W_TIME * time_eff
            + self.W_CLEAR_LAND * clear_land
            + self.W_TRACK * track
            + self.W_SAFETY_ENERGY * safety_energy
        )

    def _time_efficiency(
        self,
        ep: EpisodeMetrics,
        max_episode_steps: int,
        max_episode_time_s: float | None,
    ) -> float:
        if max_episode_time_s and max_episode_time_s > 0 and ep.completion_time is not None:
            norm = ep.completion_time / max_episode_time_s
        elif max_episode_steps > 0:
            norm = ep.steps / max_episode_steps
        else:
            return 0.0

        if norm > self.max_normalized_time:
            return 0.0
        return float(max(0.0, 1.0 - norm / self.max_normalized_time))

    def _clear_land(self, ep: EpisodeMetrics) -> float:
        ex = ep.extra or {}
        clearance = self._clearance_score(ex)
        land_stable = self._clip01(ex.get("stable_clear_rate"))
        land_impact = self._landing_impact_score(ex)
        cleared = ex.get("final_cleared_count")
        if cleared is None and "cleared_count" in ex:
            cleared = ex.get("cleared_count")
        if cleared is not None and float(cleared) > 0 and land_impact is None:
            land_impact = 0.0
        if land_impact is None:
            land_impact = 0.0
        return float(
            self.W_CLEARANCE * clearance
            + self.W_LAND_STABLE * land_stable
            + self.W_LAND_IMPACT * land_impact
        )

    def _clearance_score(self, ex: dict[str, Any]) -> float:
        samples = ex.get("apex_clearance_m")
        margin = float(self.CLEARANCE_MARGIN_M)
        if margin <= 0.0:
            return 1.0
        if isinstance(samples, (list, tuple)) and len(samples) > 0:
            scores = [self._clip01(float(c) / margin) for c in samples]
            return float(np.mean(scores))
        mean_m = ex.get("mean_apex_clearance_m")
        if mean_m is not None:
            return self._clip01(float(mean_m) / margin)
        # Legacy key (pre-raw-telemetry) — fail-closed if absent.
        legacy = ex.get("mean_apex_clearance_score")
        return self._clip01(legacy)

    def _landing_impact_score(self, ex: dict[str, Any]) -> float | None:
        samples = ex.get("landing_peak_vz")
        vz_ref = float(self.LANDING_VZ_REF)
        if isinstance(samples, (list, tuple)) and len(samples) > 0:
            if vz_ref <= 0.0:
                return 1.0
            scores = [float(max(0.0, min(1.0, 1.0 - float(v) / vz_ref))) for v in samples]
            return float(np.mean(scores))
        mean_vz = ex.get("mean_landing_peak_vz")
        if mean_vz is not None:
            if vz_ref <= 0.0:
                return 1.0
            return float(max(0.0, min(1.0, 1.0 - float(mean_vz) / vz_ref)))
        # Legacy key.
        if "mean_landing_impact_score" in ex:
            return self._clip01(ex.get("mean_landing_impact_score"))
        return None

    def _track(self, ep: EpisodeMetrics) -> float:
        """Centerline score from progress-weighted RMS |y| (fail-closed)."""
        ex = ep.extra or {}
        rms = ex.get("rms_lateral_offset_m")
        if rms is None:
            return 0.0
        y_ref = float(self.Y_REF_M)
        if y_ref <= 0.0:
            return 1.0
        ratio = float(rms) / y_ref
        return float(max(0.0, min(1.0, 1.0 - ratio * ratio)))

    def _safety_energy(self, ep: EpisodeMetrics) -> float:
        ex = ep.extra or {}
        if "mean_vertical_speed" not in ex or "mean_roll_pitch_rate" not in ex:
            return 0.0
        body_stab = self._body_stability(
            float(ex["mean_vertical_speed"]),
            float(ex["mean_roll_pitch_rate"]),
        )
        action_l2 = ex.get("mean_action_l2")
        if action_l2 is None:
            return 0.0
        energy = float(max(0.0, 1.0 - float(action_l2) / self.ACTION_L2_REF))
        return float(self.W_BODY_STAB * body_stab + self.W_ENERGY * energy)

    def _body_stability(self, mean_vert_speed: float, mean_rp_rate: float) -> float:
        vert = max(0.0, 1.0 - mean_vert_speed / self.MAX_VERTICAL_SPEED)
        rp = max(0.0, 1.0 - mean_rp_rate / self.MAX_ROLL_PITCH_RATE)
        return 0.5 * vert + 0.5 * rp

    @staticmethod
    def _clip01(raw: Any) -> float:
        if raw is None:
            return 0.0
        return float(max(0.0, min(1.0, float(raw))))
