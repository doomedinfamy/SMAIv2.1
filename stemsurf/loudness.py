"""Loudness stage: per-stem LUFS balancing and final master normalization."""

from typing import Dict

import numpy as np
import pyloudnorm as pyln

from .analysis import StemAnalysis
from .config import MixConfig


def balance_stems(stems: Dict[str, np.ndarray],
                  analyses: Dict[str, StemAnalysis],
                  cfg: MixConfig) -> Dict[str, np.ndarray]:
    """Gain each stem toward anchor_lufs + its profile offset.

    Anchor = loudest non-silent stem's target group (vocals if present).
    """
    active = {n: a for n, a in analyses.items() if not a.is_silent}
    if not active:
        return stems

    if "vocals" in active:
        anchor = active["vocals"].lufs
    else:
        anchor = max(a.lufs for a in active.values())

    out: Dict[str, np.ndarray] = {}
    for name, audio in stems.items():
        info = analyses[name]
        if info.is_silent:
            out[name] = audio
            continue
        target = anchor + cfg.profile_for(name).target_lufs_offset
        gain_db = np.clip(target - info.lufs, -12.0, 12.0)
        out[name] = (audio * 10 ** (gain_db / 20)).astype(np.float32)
    return out


def normalize_master(mix: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    meter = pyln.Meter(sr)
    try:
        current = meter.integrated_loudness(mix)
    except ValueError:
        return mix
    if not np.isfinite(current):
        return mix
    gain_db = np.clip(target_lufs - current, -24.0, 24.0)
    return (mix * 10 ** (gain_db / 20)).astype(np.float32)


def soft_limiter(mix: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Simple tanh-knee limiter to catch overs after normalization."""
    ceiling = 10 ** (ceiling_db / 20)
    peak = float(np.max(np.abs(mix)) + 1e-12)
    if peak <= ceiling:
        return mix
    return (np.tanh(mix / ceiling) * ceiling).astype(np.float32)
