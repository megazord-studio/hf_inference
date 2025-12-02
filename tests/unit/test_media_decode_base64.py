from __future__ import annotations

import base64
import io

import numpy as np
import soundfile as sf
from PIL import Image

from app.core.runners.audio.utils import (
    decode_base64_audio as decode_audio_legacy,
)
from app.core.runners.vision.utils import (
    decode_base64_image as decode_image_legacy,
)
from app.core.utils.media import decode_image_base64 as decode_image_modern


def _mk_wav_base64(sr=16000, dur=0.1, freq=440.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    raw = buf.getvalue()
    return base64.b64encode(raw).decode(
        "ascii"
    ), "data:audio/wav;base64," + base64.b64encode(raw).decode("ascii")


def _mk_png_base64(size=(16, 16), color=(120, 50, 20)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    return base64.b64encode(raw).decode(
        "ascii"
    ), "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def test_decode_audio_accepts_bare_and_data_uri():
    bare, data_uri = _mk_wav_base64()

    # Bare base64
    bio1 = decode_audio_legacy(bare)
    data1, sr1 = sf.read(bio1)
    assert sr1 in (16000, 24000, 22050)
    assert data1.size > 0

    # Data URI base64
    bio2 = decode_audio_legacy(data_uri)
    data2, sr2 = sf.read(bio2)
    assert sr2 == sr1
    assert data2.size > 0


def test_decode_image_accepts_bare_and_data_uri_legacy_and_modern():
    bare, data_uri = _mk_png_base64()

    # Legacy vision utils (used by some runners)
    img1 = decode_image_legacy(bare)
    assert img1.size == (16, 16)

    img2 = decode_image_legacy(data_uri)
    assert img2.size == (16, 16)

    # Modern shared media utils
    img3 = decode_image_modern(bare)
    assert img3.size == (16, 16)

    img4 = decode_image_modern(data_uri)
    assert img4.size == (16, 16)
