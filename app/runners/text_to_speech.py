from typing import Any
from typing import Dict

import numpy as np
from numpy.typing import NDArray
from transformers import pipeline
from transformers.pipelines import TextToAudioPipeline

from app.helpers import audio_to_bytes
from app.helpers import device_arg
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error
from app.utilities import is_no_weight_files_error


def run_tts(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run text-to-speech (text-to-audio) inference.
    Returns audio as bytes in a dictionary with metadata.
    """
    try:
        text = str(spec.get("payload", {}).get("tts_text", "")).strip()
        if not text:
            return {
                "error": "text-to-speech failed",
                "reason": "empty tts_text",
            }

        pl: TextToAudioPipeline = pipeline(
            task="text-to-audio",
            model=spec["model_id"],
            device=device_arg(dev),
        )

        out = pl(text)

        if isinstance(out, dict):
            audio_any: Any = out.get("audio")
            sr = int(out.get("sampling_rate", 16000))
        else:
            audio_any = out
            sr = 16000

        if audio_any is None:
            return {
                "error": "text-to-speech failed",
                "reason": "no audio returned",
            }

        audio_arr: NDArray[np.float32] = np.asarray(
            audio_any, dtype=np.float32
        )
        audio_bytes = audio_to_bytes(audio_arr, sr)

        return {
            "file_data": audio_bytes,
            "file_name": f"{spec['model_id'].replace('/', '_')}_tts.wav",
            "content_type": "audio/wav",
            "sampling_rate": sr,
        }
    except Exception as e:
        if is_gated_repo_error(e):
            return {
                "skipped": True,
                "reason": "gated model (no access/auth)",
                "hint": "Use facebook/mms-tts-* or request access.",
            }
        if is_missing_model_error(e) or is_no_weight_files_error(e):
            return {
                "skipped": True,
                "reason": "model not loadable (missing/unsupported weights)",
                "hint": "Use facebook/mms-tts-* or microsoft/speecht5_tts (with embeddings).",
            }
        if "speaker_embeddings" in repr(e):
            return {
                "skipped": True,
                "reason": "missing required speaker embeddings",
                "hint": "Download xvectors (Matthijs/cmu-arctic-xvectors) and pass `speaker_embeddings`.",
            }
        return {
            "error": "text-to-speech failed",
            "reason": repr(e),
            "hint": "Some models need external speaker embeddings (e.g., SpeechT5) or aren't supported by pipeline.",
        }
