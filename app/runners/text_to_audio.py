from typing import Any
from typing import Dict

from transformers import pipeline

from app.helpers import audio_to_bytes
from app.helpers import device_arg
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error
from app.utilities import is_no_weight_files_error


def run_text_to_audio(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run text-to-audio inference.
    Returns audio as bytes in a dictionary with metadata.
    """
    try:
        pl = pipeline(
            "text-to-audio", model=spec["model_id"], device=device_arg(dev)
        )
        out = pl(spec["payload"]["tta_prompt"])
        audio = (
            out["audio"] if isinstance(out, dict) and "audio" in out else out
        )
        sr = (
            out.get("sampling_rate", 32000) if isinstance(out, dict) else 32000
        )

        # Convert audio to bytes
        audio_bytes = audio_to_bytes(audio, sr)
        return {
            "file_data": audio_bytes,
            "file_name": f"{spec['model_id'].replace('/', '_')}_music.wav",
            "content_type": "audio/wav",
            "sampling_rate": sr,
        }
    except Exception as e:
        if is_gated_repo_error(e):
            return {
                "skipped": True,
                "reason": "gated model (no access/auth)",
                "hint": "Try facebook/musicgen-*",
            }
        if is_missing_model_error(e) or is_no_weight_files_error(e):
            return {
                "skipped": True,
                "reason": "model not loadable (missing/unsupported weights)",
                "hint": "Try facebook/musicgen-*",
            }
        return {"error": "text-to-audio failed", "reason": repr(e)}
