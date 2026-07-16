"""Stem separation engines.

SpleeterEngine is the default. The base class keeps the interface
pluggable so Demucs (or anything else) can be dropped in later.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import soundfile as sf


class SeparationEngine(ABC):
    """Takes a mixed track, returns {stem_name: stereo float32 array (n, 2)}."""

    @abstractmethod
    def separate(self, audio_path: str, sample_rate: int) -> Dict[str, np.ndarray]:
        ...


class SpleeterEngine(SeparationEngine):
    """Deezer Spleeter backend.

    model: 'spleeter:2stems' | 'spleeter:4stems' | 'spleeter:5stems'
    """

    def __init__(self, model: str = "spleeter:4stems"):
        self.model = model
        self._separator = None  # lazy: TF import is slow

    def _get_separator(self):
        if self._separator is None:
            from spleeter.separator import Separator
            self._separator = Separator(self.model)
        return self._separator

    def separate(self, audio_path: str, sample_rate: int) -> Dict[str, np.ndarray]:
        from spleeter.audio.adapter import AudioAdapter

        adapter = AudioAdapter.default()
        waveform, _ = adapter.load(audio_path, sample_rate=sample_rate)
        prediction = self._get_separator().separate(waveform)

        stems: Dict[str, np.ndarray] = {}
        for name, data in prediction.items():
            data = np.asarray(data, dtype=np.float32)
            if data.ndim == 1:
                data = np.stack([data, data], axis=1)
            stems[name] = data
        return stems


class FolderEngine(SeparationEngine):
    """Reads pre-separated stems from a folder of wav/flac files.

    Useful for testing the mix stage without running Spleeter, or for
    stems exported from a DAW. File names become stem names.
    """

    def __init__(self, folder: str):
        self.folder = folder

    def separate(self, audio_path: str, sample_rate: int) -> Dict[str, np.ndarray]:
        stems: Dict[str, np.ndarray] = {}
        for fname in sorted(os.listdir(self.folder)):
            base, ext = os.path.splitext(fname)
            if ext.lower() not in (".wav", ".flac", ".ogg"):
                continue
            data, sr = sf.read(os.path.join(self.folder, fname), always_2d=True)
            if sr != sample_rate:
                data = _resample(data, sr, sample_rate)
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            stems[base.lower()] = data.astype(np.float32)
        if not stems:
            raise FileNotFoundError(f"No stem audio files found in {self.folder}")
        return stems


def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(sr_in, sr_out)
    return resample_poly(data, sr_out // g, sr_in // g, axis=0)
