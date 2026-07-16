"""Shared DSP primitives: biquad filters, mid/side, panning, saturation."""

import numpy as np
from scipy.signal import sosfilt, butter


# ---------- biquad EQ (RBJ cookbook) ----------

def _biquad_sos(b0, b1, b2, a0, a1, a2) -> np.ndarray:
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def peaking_sos(fc: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    return _biquad_sos(1 + alpha * A, -2 * cw, 1 - alpha * A,
                       1 + alpha / A, -2 * cw, 1 - alpha / A)


def shelf_sos(fc: float, gain_db: float, sr: int, high: bool = True) -> np.ndarray:
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2 * np.sqrt(2)  # S = 1
    two_sqA_alpha = 2 * np.sqrt(A) * alpha
    if high:
        b0 = A * ((A + 1) + (A - 1) * cw + two_sqA_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - two_sqA_alpha)
        a0 = (A + 1) - (A - 1) * cw + two_sqA_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - two_sqA_alpha
    else:
        b0 = A * ((A + 1) - (A - 1) * cw + two_sqA_alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cw)
        b2 = A * ((A + 1) - (A - 1) * cw - two_sqA_alpha)
        a0 = (A + 1) + (A - 1) * cw + two_sqA_alpha
        a1 = -2 * ((A - 1) + (A + 1) * cw)
        a2 = (A + 1) + (A - 1) * cw - two_sqA_alpha
    return _biquad_sos(b0, b1, b2, a0, a1, a2)


def apply_sos(audio: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """audio: (n, ch). Filters each channel."""
    return sosfilt(sos, audio, axis=0).astype(np.float32)


def peaking_eq(audio: np.ndarray, fc: float, gain_db: float, q: float,
               sr: int) -> np.ndarray:
    if abs(gain_db) < 0.1:
        return audio
    return apply_sos(audio, peaking_sos(fc, gain_db, q, sr))


def high_shelf(audio: np.ndarray, fc: float, gain_db: float, sr: int) -> np.ndarray:
    if abs(gain_db) < 0.1:
        return audio
    return apply_sos(audio, shelf_sos(fc, gain_db, sr, high=True))


# ---------- stereo tools ----------

def to_mid_side(audio: np.ndarray):
    mid = (audio[:, 0] + audio[:, 1]) * 0.5
    side = (audio[:, 0] - audio[:, 1]) * 0.5
    return mid, side


def from_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def set_width(audio: np.ndarray, width: float) -> np.ndarray:
    """0 = mono, 1 = unchanged, >1 = wider."""
    mid, side = to_mid_side(audio)
    return from_mid_side(mid, side * width)


def pan(audio: np.ndarray, position: float) -> np.ndarray:
    """Constant-power pan. position in [-1, 1]."""
    theta = (position + 1.0) * np.pi / 4.0  # 0..pi/2
    gl, gr = np.cos(theta), np.sin(theta)
    out = audio.copy()
    out[:, 0] *= gl * np.sqrt(2)
    out[:, 1] *= gr * np.sqrt(2)
    return out


def mono_below(audio: np.ndarray, fc: float, sr: int) -> np.ndarray:
    """Collapse content below fc to mono (bass management)."""
    if fc <= 0:
        return audio
    mid, side = to_mid_side(audio)
    sos = butter(2, fc, btype="highpass", fs=sr, output="sos")
    side_hp = sosfilt(sos, side)
    return from_mid_side(mid, side_hp)


# ---------- saturation / excitation ----------

def soft_saturate(x: np.ndarray, drive: float = 2.0) -> np.ndarray:
    return np.tanh(x * drive) / np.tanh(drive)


def harmonic_exciter(audio: np.ndarray, fc: float, amount: float,
                     sr: int) -> np.ndarray:
    """Adds saturated high-band harmonics back into the signal.

    Classic 'aural exciter' trick to un-muffle dull stems.
    """
    if amount <= 0:
        return audio
    sos = butter(2, fc, btype="highpass", fs=sr, output="sos")
    high = sosfilt(sos, audio, axis=0)
    excited = soft_saturate(high, drive=3.0)
    return (audio + amount * excited).astype(np.float32)
