# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Spectral arc length of a hand-speed profile.

SPARC follows Balasubramanian et al., IEEE TNSRE, 2015. The value is negative.
A profile whose spectrum is concentrated at low frequency lies closer to zero
and is the smoother movement.
"""

from __future__ import annotations

import numpy as np

# Cutoff [Hz] and amplitude floor used to select the spectrum that enters the arc.
_CUTOFF_HZ = 10.0
_AMPLITUDE_THRESHOLD = 0.05
_PAD_LEVEL = 4
_MIN_SAMPLES = 8


def spectral_arc_length(speed: np.ndarray, sample_hz: float) -> float | None:
    """Return SPARC for a speed profile [m/s], or None when the profile is too short.

    Args:
        speed: Hand speed [m/s], shape [T].
        sample_hz: Sample rate [Hz].
    """
    profile = np.asarray(speed, dtype=np.float64).reshape(-1)
    if profile.size < _MIN_SAMPLES or sample_hz <= 0.0 or not np.any(profile):
        return None
    nfft = int(2 ** np.ceil(np.log2(profile.size) + _PAD_LEVEL))
    spectrum = np.abs(np.fft.rfft(profile, nfft))
    peak = float(np.max(spectrum))
    if peak <= 0.0:
        return None
    spectrum = spectrum / peak
    freq = np.fft.rfftfreq(nfft, d=1.0 / sample_hz)
    selected = freq <= _CUTOFF_HZ
    freq = freq[selected]
    spectrum = spectrum[selected]
    above = np.flatnonzero(spectrum >= _AMPLITUDE_THRESHOLD)
    if above.size < 2:
        return None
    freq = freq[above[0] : above[-1] + 1]
    spectrum = spectrum[above[0] : above[-1] + 1]
    span = float(freq[-1] - freq[0])
    if span <= 0.0:
        return None
    df = np.diff(freq) / span
    dmag = np.diff(spectrum)
    return float(-np.sum(np.sqrt(df * df + dmag * dmag)))


def sparc_from_positions(positions: np.ndarray, dt_s: float) -> float | None:
    """SPARC of the hand path. ``positions`` is [T, 3] in meters and ``dt_s`` is the sample period [s]."""
    path = np.asarray(positions, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < _MIN_SAMPLES or path.shape[1] < 3 or dt_s <= 0.0:
        return None
    velocity = np.gradient(path[:, :3], dt_s, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    return spectral_arc_length(speed, 1.0 / dt_s)
