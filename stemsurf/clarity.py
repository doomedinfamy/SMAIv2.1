
from typing import Dict

import numpy as np

from .analysis import StemAnalysis
from .config import MixConfig
from . import dsp


def _pocket_center_q(low: float, high: float):
    fc = float(np.sqrt(low * high))  # geometric center
    bw_oct = np.log2(high / low)
    q = max(0.5, 1.414 / max(bw_oct, 0.1))
    return fc, q


def carve_pockets(stems: Dict[str, np.ndarray], cfg: MixConfig,
                  sr: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    names = list(stems.keys())
    for name in names:
        audio = stems[name]
        prof = cfg.profile_for(name)
        fc, q = _pocket_center_q(*prof.pocket)
        # gentle boost in own pocket
        audio = dsp.peaking_eq(audio, fc, +1.5, q, sr)
        # dip inside every other stem's pocket
        for other in names:
            if other == name:
                continue
            oprof = cfg.profile_for(other)
            ofc, oq = _pocket_center_q(*oprof.pocket)
            audio = dsp.peaking_eq(audio, ofc, -oprof.carve_db, oq, sr)
        out[name] = audio
    return out


def demuffle(audio: np.ndarray, info: StemAnalysis, cfg: MixConfig,
             sr: int) -> np.ndarray:
    """Brighten a dull stem proportionally to how muffled it is."""
    if info.is_silent or not info.is_muffled:
        return audio

    # 0 (barely muffled) .. 1 (no HF at all)
    dullness = 1.0 - min(info.hf_ratio / cfg.muffle_threshold, 1.0)
    boost_db = dullness * cfg.max_brighten_db

    audio = dsp.high_shelf(audio, 5000.0, boost_db, sr)
    audio = dsp.harmonic_exciter(audio, 3000.0, cfg.exciter_amount * dullness, sr)
    return audio
