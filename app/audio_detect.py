from __future__ import annotations

import math
import os
import subprocess
import tempfile
from typing import List

import numpy as np

from .settings import settings


WHISTLE_DB = 5.5
CROWD_DECAY_SEC = 2.0


def _ffmpeg_to_wav(src: str, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-i",
                src,
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-f",
                "wav",
                tmp.name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    return tmp.name


def whistle_crowd_spikes(src: str) -> List[float]:
    if not settings.AUDIO_ENABLE:
        return []

    try:
        wav_path = _ffmpeg_to_wav(src, settings.AUDIO_SR)
    except Exception:
        return []

    try:
        import soundfile as sf  # type: ignore

        samples, sr = sf.read(wav_path, dtype="float32")
    except Exception:
        return []
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    if getattr(samples, "size", 0) == 0:
        return []

    total_len = len(samples)
    win = max(1, min(total_len, int(sr * 0.032)))
    hop = max(1, int(sr * 0.016))
    if win <= 0 or total_len < win:
        return []

    freqs = np.fft.rfftfreq(win, 1 / sr)
    f_lo, f_hi = settings.AUDIO_WHISTLE_BAND
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band_mask):
        return []

    window = np.hanning(win)
    limit = max(0, total_len - win)
    energies: List[float] = []
    for idx in range(0, limit + 1, hop):
        segment = samples[idx : idx + win]
        if len(segment) < win:
            continue
        spec = np.fft.rfft(segment * window)
        band_energy = np.abs(spec[band_mask]) ** 2
        energy = float(np.sum(band_energy) + 1e-9)
        energies.append(10.0 * math.log10(energy))

    if not energies:
        return []

    median = float(np.median(energies))
    spikes: List[float] = []
    threshold_db = min(float(settings.AUDIO_MIN_SPIKE_DB), WHISTLE_DB)
    last_low_time = float("-inf")
    for i, energy in enumerate(energies):
        center = (i * hop + win // 2) / sr
        if energy <= median:
            last_low_time = center
        if energy - median < threshold_db:
            continue
        if (center - last_low_time) < CROWD_DECAY_SEC:
            continue
        if not spikes or (center - spikes[-1]) >= settings.AUDIO_MIN_GAP_SEC:
            spikes.append(center)

    return spikes
