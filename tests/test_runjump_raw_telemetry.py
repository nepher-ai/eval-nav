# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Offline unit tests for RunJump raw telemetry derivation + scorer maps."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

# Load modules without importing the full eval_nav package (avoids isaaclab).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _ensure_pkg(name: str, path: Path | None = None) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    if path is not None:
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


def _load():
    import importlib.util

    _ensure_pkg("eval_nav", _ROOT / "eval_nav")
    _ensure_pkg("eval_nav.core", _ROOT / "eval_nav" / "core")
    _ensure_pkg("eval_nav.core.telemetry", _ROOT / "eval_nav" / "core" / "telemetry")
    _ensure_pkg(
        "eval_nav.core.telemetry.derive",
        _ROOT / "eval_nav" / "core" / "telemetry" / "derive",
    )
    _ensure_pkg("eval_nav.domain", _ROOT / "eval_nav" / "domain")
    _ensure_pkg("eval_nav.core.scorers", _ROOT / "eval_nav" / "core" / "scorers")
    _ensure_pkg(
        "eval_nav.core.scorers.navigation",
        _ROOT / "eval_nav" / "core" / "scorers" / "navigation",
    )

    def load(fullname: str, rel: str):
        path = _ROOT / "eval_nav" / rel
        spec = importlib.util.spec_from_file_location(fullname, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = mod
        spec.loader.exec_module(mod)
        return mod

    metrics = load("eval_nav.domain.metrics", "domain/metrics.py")
    base = load("eval_nav.core.scorers.base", "core/scorers/base.py")
    kin = load(
        "eval_nav.core.telemetry.derive.kinematics",
        "core/telemetry/derive/kinematics.py",
    )
    rj = load(
        "eval_nav.core.telemetry.derive.runjump",
        "core/telemetry/derive/runjump.py",
    )
    # Wire relative imports used by the scorers.
    sys.modules["eval_nav.core.telemetry.derive.runjump"] = rj
    nat = load(
        "eval_nav.core.telemetry.derive.naturalness",
        "core/telemetry/derive/naturalness.py",
    )
    sys.modules["eval_nav.core.telemetry.derive.naturalness"] = nat
    scorer_mod = load(
        "eval_nav.core.scorers.navigation.humanoid_runjump",
        "core/scorers/navigation/humanoid_runjump.py",
    )
    # Parent must be importable for the v2 subclass.
    sys.modules["eval_nav.core.scorers.navigation.humanoid_runjump"] = scorer_mod
    scorer_v2_mod = load(
        "eval_nav.core.scorers.navigation.humanoid_runjump_v2",
        "core/scorers/navigation/humanoid_runjump_v2.py",
    )
    return metrics, kin, rj, nat, scorer_mod, scorer_v2_mod


def _synthetic_series(t: int = 200, *, crossed: bool = False, yaw_rad: float = 0.0) -> tuple[dict[str, np.ndarray], dict]:
    """Build a synthetic straight-course jump with known clearance / impact."""
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    xs = [10.0]
    hs = [0.40]
    ts = [0.20]

    root_pos = np.zeros((t, 3), dtype=np.float32)
    root_lin_b = np.zeros((t, 3), dtype=np.float32)
    root_lin_w = np.zeros((t, 3), dtype=np.float32)
    root_ang_b = np.zeros((t, 3), dtype=np.float32)
    ankle = np.zeros((t, 2, 3), dtype=np.float32)
    forces = np.zeros((t, 2, 3), dtype=np.float32)
    grav = np.zeros((t, 3), dtype=np.float32)
    grav[:, 2] = -1.0
    mode = np.zeros(t, dtype=np.int64)
    hl_action = np.zeros((t, 3), dtype=np.float32)
    obs_idx = np.zeros(t, dtype=np.int64)
    cleared = np.zeros(t, dtype=np.int64)

    # Cruise along +x at 2 m/s (dt=0.02 → 0.04 m/step); need >10.25 m to clear.
    # 200 steps * 0.04 = 8 m is too short — use 3.5 m/s → 0.07 m/step → 14 m.
    for i in range(t):
        s = min(3.5 * i * 0.02, 20.0)
        root_pos[i, 0] = s
        root_pos[i, 2] = 0.75
        root_lin_b[i, 0] = 3.5
        root_lin_w[i, 0] = 3.5
        ankle[i, :, 2] = 0.05
        forces[i, :, 2] = 50.0  # both feet planted
        hl_action[i, 0] = 0.5

    # Jump window near the hurdle (s≈10 at step ~143): mode JUMP, apex ankle z = 0.55
    # clearance = 0.55 - 0.40 = 0.15 → score 1.0 at margin 0.15
    for i in range(120, 151):
        mode[i] = 1
        forces[i] = 0.0
        ankle[i, :, 2] = 0.05 + 0.50 * np.sin(np.pi * (i - 120) / 30.0)
        root_lin_w[i, 2] = 1.0 if i < 135 else -2.0  # down after apex

    # Dual-foot touchdown rising edge at step 151 while still JUMP briefly,
    # then RUN. Peak -vz during window should be ~2.0.
    mode[151] = 1
    forces[151, :, 2] = 80.0
    root_lin_w[151, 2] = -2.0
    for i in range(152, 160):
        mode[i] = 0
        forces[i, :, 2] = 80.0
        root_lin_w[i, 2] = -1.0

    # Cross back face at s > 10.2 → around step where s exceeds 10.25
    # After crossing, stay upright for stability probe.
    for i in range(t):
        if root_pos[i, 0] > 10.25:
            cleared[i] = 1
            obs_idx[i] = 1
        else:
            obs_idx[i] = 0

    # Nominal stance: left at +y, right at -y (uncrossed).
    ankle[:, 0, 1] = 0.12
    ankle[:, 1, 1] = -0.12
    if crossed:
        # Invert lateral sides during JUMP window -> X-legs.
        for i in range(120, 152):
            ankle[i, 0, 1] = -0.20
            ankle[i, 1, 1] = 0.20

    # Identity quat (wxyz); optional yaw rotation of the whole body frame.
    root_quat = np.zeros((t, 4), dtype=np.float32)
    root_quat[:, 0] = np.cos(0.5 * yaw_rad)
    root_quat[:, 3] = np.sin(0.5 * yaw_rad)
    if abs(yaw_rad) > 1e-9:
        # Rotate ankle XY about origin by yaw so world positions match body yaw.
        c, s_ = np.cos(yaw_rad), np.sin(yaw_rad)
        for side in (0, 1):
            y = ankle[:, side, 1].copy()
            x = ankle[:, side, 0].copy()
            # ankles currently have x~0 relative; place relative to root then rotate
            pass
        # Build ankle world XY from body-frame offsets rotated by yaw.
        for i in range(t):
            for side, y_body in ((0, float(ankle[i, 0, 1])), (1, float(ankle[i, 1, 1]))):
                # body offset (0, y_body) -> world
                ankle[i, side, 0] = root_pos[i, 0] + (-s_ * y_body)
                ankle[i, side, 1] = root_pos[i, 1] + (c * y_body)
    else:
        for i in range(t):
            ankle[i, 0, 0] = root_pos[i, 0]
            ankle[i, 1, 0] = root_pos[i, 0]
            ankle[i, 0, 1] = root_pos[i, 1] + float(ankle[i, 0, 1])
            ankle[i, 1, 1] = root_pos[i, 1] + float(ankle[i, 1, 1])

    series = {
        "root_pos_w": root_pos,
        "root_quat_w": root_quat,
        "root_lin_vel_b": root_lin_b,
        "root_lin_vel_w": root_lin_w,
        "root_ang_vel_b": root_ang_b,
        "ankle_pos_w": ankle,
        "foot_contact_force_w": forces,
        "projected_gravity_b": grav,
        "hl_mode": mode,
        "hl_action": hl_action,
        "obstacle_index": obs_idx,
        "cleared_count": cleared,
    }
    meta = {
        "env_origin": origin,
        "obstacle_xs": xs,
        "obstacle_hs": hs,
        "obstacle_ts": ts,
        "num_obstacles": 1,
        "step_dt": 0.02,
        "path_length": 20.0,
    }
    return series, meta


def test_kinematics_and_track():
    metrics, kin, rj, _nat, scorer_mod, _v2 = _load()
    series, meta = _synthetic_series()
    # Add a small lateral wander mid-course.
    series["root_pos_w"][50:60, 1] = 0.3

    extra = kin.derive_kinematics_episode(series, meta)
    assert "mean_speed" in extra
    assert extra["mean_speed"] > 0.0
    assert "rms_lateral_offset_m" in extra
    assert extra["rms_lateral_offset_m"] >= 0.0
    assert "mean_action_l2" in extra
    print("kinematics ok", {k: round(v, 4) if isinstance(v, float) else v for k, v in extra.items()})


def test_runjump_events():
    _, kin, rj, _nat, scorer_mod, _v2 = _load()
    series, meta = _synthetic_series()
    events = rj.derive_runjump_episode(series, meta)
    assert len(events["apex_clearance_m"]) >= 1
    # Peak ankle ≈ 0.55, hurdle 0.40 → clearance ≈ 0.15
    assert abs(events["apex_clearance_m"][0] - 0.15) < 0.05, events["apex_clearance_m"]
    assert len(events["landing_peak_vz"]) >= 1
    assert events["landing_peak_vz"][0] > 0.5
    assert events["crossings"] >= 1.0
    assert events["final_cleared_count"] >= 1.0
    print(
        "runjump ok",
        {
            "apex": events["apex_clearance_m"],
            "vz": events["landing_peak_vz"],
            "crossings": events["crossings"],
            "stable": events["stable_crossings"],
            "rate": events["stable_clear_rate"],
        },
    )


def test_scorer_maps_raw_extras():
    metrics_mod, kin, rj, _nat, scorer_mod, _v2 = _load()
    series, meta = _synthetic_series()
    extra = kin.derive_kinematics_episode(series, meta)
    extra.update(rj.derive_runjump_episode(series, meta))
    extra["step_dt"] = 0.02

    scorer = scorer_mod.HumanoidRunJumpScorer()
    assert scorer.CLEARANCE_MARGIN_M == 0.15
    assert scorer.LANDING_VZ_REF == 3.5
    d = scorer.to_dict()
    assert "derivation" in d
    assert d["thresholds"]["clearance_margin_m"] == 0.15

    ep = metrics_mod.EpisodeMetrics(
        episode_id=0,
        scene=0,
        seed=0,
        success=True,
        steps=200,
        timeout=False,
        env_id="humanoid-runjump-course-v1",
        completion_time=4.0,
        extra=extra,
    )
    agg = metrics_mod.AggregateMetrics.from_episodes([ep])
    score = scorer.compute_score(agg, 2000, [ep], max_episode_time_s=40.0)
    assert 0.0 < score <= 1.0, score

    # Perfect centered track with zero lateral offset.
    extra2 = dict(extra)
    extra2["rms_lateral_offset_m"] = 0.0
    ep2 = metrics_mod.EpisodeMetrics(
        episode_id=1,
        scene=0,
        seed=0,
        success=True,
        steps=200,
        timeout=False,
        completion_time=4.0,
        extra=extra2,
    )
    track = scorer._track(ep2)
    assert abs(track - 1.0) < 1e-9

    # Clearance map: 0.15 m / 0.15 margin → 1.0
    c = scorer._clearance_score({"apex_clearance_m": [0.15]})
    assert abs(c - 1.0) < 1e-9
    # Impact: vz=0 → 1.0; vz=3.5 → 0.0
    assert abs(scorer._landing_impact_score({"landing_peak_vz": [0.0]}) - 1.0) < 1e-9
    assert abs(scorer._landing_impact_score({"landing_peak_vz": [3.5]}) - 0.0) < 1e-9
    print("scorer ok", round(score, 4), "track", track)


def test_legacy_keys_still_accepted():
    """Fail-closed legacy path for old logged extras."""
    metrics_mod, _, _, _nat, scorer_mod, _v2 = _load()
    scorer = scorer_mod.HumanoidRunJumpScorer()
    ep = metrics_mod.EpisodeMetrics(
        episode_id=0,
        scene=0,
        seed=0,
        success=True,
        steps=100,
        timeout=False,
        completion_time=10.0,
        extra={
            "mean_apex_clearance_score": 1.0,
            "stable_clear_rate": 1.0,
            "mean_landing_impact_score": 1.0,
            "rms_lateral_offset_m": 0.0,
            "mean_vertical_speed": 0.0,
            "mean_roll_pitch_rate": 0.0,
            "mean_action_l2": 0.0,
            "final_cleared_count": 1.0,
        },
    )
    agg = metrics_mod.AggregateMetrics.from_episodes([ep])
    score = scorer.compute_score(agg, 2000, [ep], max_episode_time_s=40.0)
    # time_eff = 1 - 10/40 = 0.75
    # Q = 0.35*0.75 + 0.30*1 + 0.20*1 + 0.15*1 = 0.2625+0.65 = 0.9125
    # score = 0.25 + 0.75*0.9125 = 0.934375
    assert abs(score - 0.934375) < 1e-5, score
    print("legacy ok", score)




def test_naturalness_clean_is_one():
    """Deadbands leave a clean upright uncrossed run at N == 1.0."""
    metrics_mod, kin, rj, nat, scorer_mod, scorer_v2_mod = _load()
    series, meta = _synthetic_series(crossed=False)
    extra = kin.derive_kinematics_episode(series, meta)
    extra.update(rj.derive_runjump_episode(series, meta))
    extra.update(nat.derive_naturalness_episode(series, meta))
    extra["course_length_m"] = 20.0
    extra["step_dt"] = 0.02

    assert extra["leg_cross_depth_m"] == 0.0
    assert extra["leg_cross_time_frac"] == 0.0
    assert extra["lateral_tilt_deg"] < 25.0
    assert extra["tilt_deg"] < 60.0

    ep = metrics_mod.EpisodeMetrics(
        episode_id=0, scene=0, seed=0, success=True, steps=200, timeout=False,
        completion_time=20.0 / 2.25, extra=extra,
    )
    scorer = scorer_v2_mod.HumanoidRunJumpScorerV2()
    n = scorer._naturalness(ep)
    assert abs(n - 1.0) < 1e-9, n
    print("naturalness clean ok", n)


def test_naturalness_xlegs_zeroes_v2_not_v1():
    """Sustained X-legs drives v2 to ~0 while v1 on the same extras is unchanged."""
    metrics_mod, kin, rj, nat, scorer_mod, scorer_v2_mod = _load()
    series, meta = _synthetic_series(crossed=True)
    extra = kin.derive_kinematics_episode(series, meta)
    extra.update(rj.derive_runjump_episode(series, meta))
    extra.update(nat.derive_naturalness_episode(series, meta))
    extra["course_length_m"] = 20.0
    extra["step_dt"] = 0.02
    # Keep body rates inside spin deadband so only crossing fires.
    extra["mean_roll_pitch_rate"] = 0.0
    extra["mean_yaw_rate"] = 0.0

    assert extra["leg_cross_depth_m"] >= 0.30 - 1e-6

    ep = metrics_mod.EpisodeMetrics(
        episode_id=0, scene=0, seed=0, success=True, steps=200, timeout=False,
        completion_time=8.0, extra=extra,
    )
    agg = metrics_mod.AggregateMetrics.from_episodes([ep])

    v1 = scorer_mod.HumanoidRunJumpScorer()
    v2 = scorer_v2_mod.HumanoidRunJumpScorerV2()
    score_v1 = v1.compute_score(agg, 2000, [ep], max_episode_time_s=40.0)
    score_v2 = v2.compute_score(agg, 2000, [ep], max_episode_time_s=40.0)
    n = v2._naturalness(ep)
    assert n < 1e-6, n
    assert score_v2 < 1e-6, score_v2
    assert score_v1 > 0.5, score_v1
    print("xlegs ok", "v1", round(score_v1, 4), "v2", score_v2, "N", n)


def test_naturalness_yawed_root_matches():
    """Crossing detection is yaw-invariant (30 deg root yaw)."""
    _, _, _, nat, _, _ = _load()
    series0, meta = _synthetic_series(crossed=True, yaw_rad=0.0)
    series1, _ = _synthetic_series(crossed=True, yaw_rad=np.deg2rad(30.0))
    d0 = nat.derive_naturalness_episode(series0, meta)
    d1 = nat.derive_naturalness_episode(series1, meta)
    assert abs(d0["leg_cross_depth_m"] - d1["leg_cross_depth_m"]) < 0.02, (d0, d1)
    print("yaw invariance ok", d0["leg_cross_depth_m"], d1["leg_cross_depth_m"])


def test_speed_term_length_invariant_and_cap():
    """Same m/s on 20 m and 55 m courses score identically; 3 m/s caps at 1.0."""
    metrics_mod, _, _, _, _, scorer_v2_mod = _load()
    scorer = scorer_v2_mod.HumanoidRunJumpScorerV2()

    def ep_at(length_m: float, speed_mps: float) -> object:
        elapsed = length_m / speed_mps
        return metrics_mod.EpisodeMetrics(
            episode_id=0, scene=0, seed=0, success=True,
            steps=int(round(elapsed / 0.02)), timeout=False,
            completion_time=elapsed,
            extra={
                "course_length_m": length_m,
                "step_dt": 0.02,
                # Naturalness keys at perfect values so N=1 if used.
                "leg_cross_depth_m": 0.0,
                "leg_cross_time_frac": 0.0,
                "lateral_tilt_deg": 0.0,
                "tilt_deg": 0.0,
                "mean_roll_pitch_rate": 0.0,
                "mean_yaw_rate": 0.0,
                "apex_clearance_m": [0.15],
                "stable_clear_rate": 1.0,
                "landing_peak_vz": [0.0],
                "rms_lateral_offset_m": 0.0,
            },
        )

    e20 = ep_at(20.0, 2.25)
    e55 = ep_at(55.0, 2.25)
    s20 = scorer._speed_score(e20, 2000, 40.0)
    s55 = scorer._speed_score(e55, 2000, 40.0)
    assert abs(s20 - s55) < 1e-9, (s20, s55)
    assert abs(s20 - (2.25 / 3.0)) < 1e-9, s20

    e_fast = ep_at(20.0, 4.0)
    assert abs(scorer._speed_score(e_fast, 2000, 40.0) - 1.0) < 1e-9
    print("speed term ok", s20)


def test_v2_hand_computed_score():
    """Hand-computed v2 score on a clean synthetic episode."""
    metrics_mod, kin, rj, nat, _, scorer_v2_mod = _load()
    series, meta = _synthetic_series(crossed=False)
    extra = kin.derive_kinematics_episode(series, meta)
    extra.update(rj.derive_runjump_episode(series, meta))
    extra.update(nat.derive_naturalness_episode(series, meta))
    extra["course_length_m"] = 20.0
    extra["step_dt"] = 0.02
    # Force perfect landing/track/rates for a deterministic hand check.
    extra["stable_clear_rate"] = 1.0
    extra["landing_peak_vz"] = [0.0]
    extra["rms_lateral_offset_m"] = 0.0
    extra["mean_roll_pitch_rate"] = 0.0
    extra["mean_yaw_rate"] = 0.0
    extra["apex_clearance_m"] = [0.15]

    # 20 m in ~5.714 s at 3.5 m/s cruise -> speed = min(3.5/3, 1) = 1.0
    ep = metrics_mod.EpisodeMetrics(
        episode_id=0, scene=0, seed=0, success=True, steps=200, timeout=False,
        completion_time=20.0 / 3.5, extra=extra,
    )
    agg = metrics_mod.AggregateMetrics.from_episodes([ep])
    scorer = scorer_v2_mod.HumanoidRunJumpScorerV2()
    score = scorer.compute_score(agg, 2000, [ep], max_episode_time_s=40.0)

    # P = 0.40*1 + 0.20*1 + 0.15*1 + 0.15*1 + 0.10*1 = 1.0; N = 1; SR = 1
    assert abs(score - 1.0) < 1e-6, score
    print("v2 hand score ok", score)


if __name__ == "__main__":
    test_kinematics_and_track()
    test_runjump_events()
    test_scorer_maps_raw_extras()
    test_legacy_keys_still_accepted()
    test_naturalness_clean_is_one()
    test_naturalness_xlegs_zeroes_v2_not_v1()
    test_naturalness_yawed_root_matches()
    test_speed_term_length_invariant_and_cap()
    test_v2_hand_computed_score()
    print("ALL PASSED")
