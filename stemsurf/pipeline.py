

import os
from typing import Dict, Optional

import numpy as np
import soundfile as sf

from .analysis import StemAnalysis, analyze
from .clarity import carve_pockets, demuffle
from .config import MixConfig
from .loudness import balance_stems, normalize_master, soft_limiter
from .separation import SeparationEngine, SpleeterEngine
from .spacing import space_stem


class MixPipeline:
    def __init__(self, engine: Optional[SeparationEngine] = None,
                 config: Optional[MixConfig] = None):
        self.engine = engine or SpleeterEngine()
        self.cfg = config or MixConfig()

    def process(self, input_path: str, output_dir: str,
                export_stems: bool = True, verbose: bool = True) -> str:
        cfg = self.cfg
        os.makedirs(output_dir, exist_ok=True)
        log = print if verbose else (lambda *a, **k: None)

        log(f"[1/6] Separating stems: {os.path.basename(input_path)}")
        stems = self.engine.separate(input_path, cfg.sample_rate)
        log(f"      -> {', '.join(stems)}")

        log("[2/6] Analyzing stems")
        analyses: Dict[str, StemAnalysis] = {}
        for name, audio in stems.items():
            info = analyze(audio, cfg.sample_rate, cfg.muffle_threshold)
            analyses[name] = info
            flags = []
            if info.is_silent:
                flags.append("silent")
            if info.is_muffled:
                flags.append("muffled")
            log(f"      {name:8s} {info.lufs:7.1f} LUFS  "
                f"centroid {info.spectral_centroid:6.0f} Hz  "
                f"HF {info.hf_ratio:.3f} {' '.join(flags)}")

        log("[3/6] Carving frequency pockets")
        stems = carve_pockets(stems, cfg, cfg.sample_rate)

        log("[4/6] Restoring clarity on muffled stems")
        stems = {
            name: demuffle(audio, analyses[name], cfg, cfg.sample_rate)
            for name, audio in stems.items()
        }

        log("[5/6] Stereo spacing + loudness balance")
        stems = {
            name: space_stem(audio, cfg.profile_for(name), cfg.sample_rate)
            for name, audio in stems.items()
        }
        stems = balance_stems(stems, analyses, cfg)

        log("[6/6] Mixdown, normalize, limit")
        n = max(a.shape[0] for a in stems.values())
        mix = np.zeros((n, 2), dtype=np.float32)
        for audio in stems.values():
            mix[: audio.shape[0]] += audio
        mix = normalize_master(mix, cfg.sample_rate, cfg.target_lufs)
        mix = soft_limiter(mix, cfg.limiter_ceiling_db)

        base = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(output_dir, f"{base}_remix.wav")
        sf.write(out_path, mix, cfg.sample_rate)

        if export_stems:
            stem_dir = os.path.join(output_dir, f"{base}_stems")
            os.makedirs(stem_dir, exist_ok=True)
            for name, audio in stems.items():
                sf.write(os.path.join(stem_dir, f"{name}.wav"),
                         soft_limiter(audio, cfg.limiter_ceiling_db),
                         cfg.sample_rate)
            log(f"      stems -> {stem_dir}")

        log(f"      mix   -> {out_path}")
        return out_path
