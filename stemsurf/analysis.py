"""Per-stem analysis: loudness, spectral balance, muffle detection."""

from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln


@dataclass
class StemAnalysis:
    lufs: float               # integrated loudness
    rms_db: float
    spectral_centroid: float  # Hz — where the energy "sits"
    hf_ratio: float           # energy above 4 kHz / total energy
    is_muffled: bool
    is_silent: bool


def analyze(audio: np.ndarray, sr: int, muffle_threshold: float = 0.08) -> StemAnalysis:
    mono = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-12)
    rms_db = 20 * np.log10(rms)
    is_silent = rms_db < -60.0

    if is_silent:
        return StemAnalysis(-np.inf, rms_db, 0.0, 0.0, False, True)

    # Integrated loudness (BS.1770)
    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(audio))
    except ValueError:  # too short
        lufs = rms_db

    # Spectral features from an averaged magnitude spectrum
    n = min(len(mono), sr * 30)  # analyze up to 30 s
    seg = mono[:n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    energy = mag ** 2
    total = float(energy.sum() + 1e-12)

    centroid = float((freqs * energy).sum() / total)
    hf_ratio = float(energy[freqs >= 4000.0].sum() / total)

    return StemAnalysis(
        lufs=lufs,
        rms_db=rms_db,
        spectral_centroid=centroid,
        hf_ratio=hf_ratio,
        is_muffled=hf_ratio < muffle_threshold,
        is_silent=False,
    )
