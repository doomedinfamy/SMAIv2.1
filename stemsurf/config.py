

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class StemProfile:
    pan: float = 0.0
    width: float = 1.0
    pocket: Tuple[float, float] = (200.0, 4000.0)
    carve_db: float = 2.0
    target_lufs_offset: float = 0.0
    keep_low_mono_below: float = 120.0  # bass management: mono the lows


# Profiles for Spleeter's 4/5-stem models.
STEM_PROFILES: Dict[str, StemProfile] = {
    "vocals": StemProfile(
        pan=0.0, width=1.1, pocket=(900.0, 4500.0), carve_db=3.0,
        target_lufs_offset=0.0,
    ),
    "drums": StemProfile(
        pan=0.0, width=1.4, pocket=(60.0, 200.0), carve_db=2.0,
        target_lufs_offset=-1.5,
    ),
    "bass": StemProfile(
        pan=0.0, width=0.3, pocket=(40.0, 250.0), carve_db=3.0,
        target_lufs_offset=-3.0,
    ),
    "piano": StemProfile(
        pan=-0.25, width=1.2, pocket=(250.0, 1200.0), carve_db=1.5,
        target_lufs_offset=-4.0,
    ),
    "other": StemProfile(
        pan=0.15, width=1.5, pocket=(300.0, 2000.0), carve_db=1.5,
        target_lufs_offset=-4.5,
    ),
}


@dataclass
class MixConfig:
    sample_rate: int = 44100
    target_lufs: float = -14.0          # streaming-standard master loudness
    muffle_threshold: float = 0.08      # HF-energy ratio below this = muffled
    max_brighten_db: float = 6.0        # cap on clarity high-shelf boost
    exciter_amount: float = 0.15        # harmonic excitation mix (0..1)
    limiter_ceiling_db: float = -1.0    # true-peak-ish ceiling
    profiles: Dict[str, StemProfile] = field(
        default_factory=lambda: dict(STEM_PROFILES)
    )

    def profile_for(self, stem_name: str) -> StemProfile:
        return self.profiles.get(stem_name, StemProfile())
