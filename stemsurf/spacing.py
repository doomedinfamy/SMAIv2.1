"""Stereo spacing: places each stem in the stereo field per its profile."""

import numpy as np

from .config import StemProfile
from . import dsp


def space_stem(audio: np.ndarray, profile: StemProfile, sr: int) -> np.ndarray:
    out = dsp.set_width(audio, profile.width)
    out = dsp.pan(out, profile.pan)
    out = dsp.mono_below(out, profile.keep_low_mono_below, sr)
    return out.astype(np.float32)
