"""Shared utilities for audio runners."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any
from typing import Tuple

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

# Single definition of np to avoid redefinition warning
np: Any
try:
    import numpy as _np

    np = _np
except Exception:  # pragma: no cover
    np = None

log = logging.getLogger("app.runners.audio")


def decode_base64_audio(b64: str) -> io.BytesIO:
    """Decode base64 audio to BytesIO.

    Accepts both full data URIs (e.g., "data:audio/wav;base64,....") and
    bare base64 payloads ("AAAA...").
    """
    try:
        payload = b64.split(",", 1)[1] if "," in b64 else b64
        audio_bytes = base64.b64decode(payload)
        return io.BytesIO(audio_bytes)
    except Exception:
        # Best-effort: attempt to decode as-is
        return io.BytesIO(base64.b64decode(b64))


def resample_audio(audio: Any, sr: int, target_sr: int) -> Tuple[Any, int]:
    """Resample audio to target sample rate using linear interpolation.

    Returns (audio_data, sample_rate). If numpy is unavailable or parameters are
    invalid, returns original audio unchanged.
    """
    if target_sr is None or sr == target_sr or sr <= 0:
        return audio, sr
    if np is None:  # numpy unavailable
        return audio, sr

    ratio = target_sr / sr
    new_len = int(len(audio) * ratio) if hasattr(audio, "__len__") else 0
    if new_len < 2 or new_len >= 10_000_000:
        return audio, sr

    try:
        x_old = np.linspace(
            0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float32
        )
        x_new = np.linspace(
            0.0, 1.0, num=new_len, endpoint=False, dtype=np.float32
        )
        resampled = np.interp(x_new, x_old, audio).astype(np.float32)
        return resampled, target_sr
    except Exception as e:
        log.warning("audio resample failed; continuing with original: %s", e)
        return audio, sr


def normalize_audio(data: Any) -> Any:
    """Normalize audio to mono float32. If numpy unavailable, return input as-is."""
    if np is None:
        return data
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except Exception:
            return data

    # Convert to mono heuristics
    if hasattr(data, "ndim") and data.ndim == 2:
        if data.shape[0] <= 8 and data.shape[0] <= data.shape[1]:
            data = data[0]
        elif data.shape[1] <= 8 and data.shape[1] < data.shape[0]:
            data = data[:, 0]
        else:
            data = data[0]
    elif hasattr(data, "ndim") and data.ndim > 2:
        data = data.reshape(-1)

    # Convert to float
    if hasattr(data, "dtype") and not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    return data


def get_target_sample_rate(processor: Any) -> int:
    """Get target sample rate from processor or fallback to 16000."""
    target_sr = getattr(processor, "sampling_rate", None)
    if target_sr is None and hasattr(processor, "feature_extractor"):
        target_sr = getattr(processor.feature_extractor, "sampling_rate", None)
    return int(target_sr) if target_sr is not None else 16000
