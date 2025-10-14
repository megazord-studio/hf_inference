import os
from typing import Any
from typing import Dict

from transformers import pipeline

from app.helpers import device_arg
from app.helpers import get_upload_file_path
from app.helpers import safe_json
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error
from app.utilities import is_no_weight_files_error


def run_zero_shot_audio_classification(
    spec: RunnerSpec, dev: str
) -> Dict[str, Any]:
    """
    Run zero-shot audio classification inference.
    Accepts either audio_path or UploadFile from spec["files"]["audio"].
    Returns the result as a dictionary instead of printing.
    """
    audio_file = spec.get("files", {}).get("audio")

    if audio_file is not None:
        # Save temporarily for pipeline
        temp_path = f"/tmp/audio_{os.getpid()}.wav"
        saved_path = get_upload_file_path(audio_file, temp_path)
        if not saved_path:
            return {
                "error": "zero-shot-audio-classification failed",
                "reason": "failed to persist uploaded audio",
            }
        audio_path = saved_path
    else:
        ap = spec.get("payload", {}).get("audio_path")
        audio_path = ap if isinstance(ap, str) and ap else "audio.wav"

    try:
        if audio_path is None:
            return {
                "error": "zero-shot-audio-classification failed",
                "reason": "audio path resolution failed",
            }

        p: Dict[str, Any] = spec.get("payload", {})
        labels_any: Any = p.get("candidate_labels", [])
        if not isinstance(labels_any, (list, tuple)) or not labels_any:
            return {
                "error": "zero-shot-audio-classification failed",
                "reason": "candidate_labels must be a non-empty list",
            }
        candidate_labels: list[str] = [str(x) for x in labels_any]

        pl = pipeline(
            "zero-shot-audio-classification",
            model=spec["model_id"],
            device=device_arg(dev),
        )
        out = pl(audio_path, candidate_labels=candidate_labels)
        for o in out:
            o["score"] = float(o["score"])
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e) or is_no_weight_files_error(e):
            return {
                "skipped": True,
                "reason": "model not loadable (missing/unsupported weights)",
            }
        return {
            "error": "zero-shot-audio-classification failed",
            "reason": repr(e),
            "hint": "May require torchaudio/librosa & proper CUDA vision deps.",
        }
    finally:
        # Cleanup temp file
        if (
            audio_file is not None
            and audio_path is not None
            and os.path.exists(audio_path)
        ):
            try:
                os.remove(audio_path)
            except Exception:
                pass
