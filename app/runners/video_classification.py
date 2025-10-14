import os
from typing import Any
from typing import Dict

from transformers import pipeline
from transformers.pipelines import VideoClassificationPipeline

from app.helpers import device_arg
from app.helpers import get_upload_file_path
from app.helpers import safe_json
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def run_video_classification(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run video classification inference.
    Accepts either video_path or UploadFile from spec['files']['video'].
    Returns the result as a dictionary instead of printing.
    """
    # Handle UploadFile or fallback to path
    video_file = spec.get("files", {}).get("video")
    video_path: str

    if video_file is not None:
        # Save temporarily for pipeline
        temp_path = f"/tmp/video_{os.getpid()}.mp4"
        saved_path = get_upload_file_path(video_file, temp_path)
        if not saved_path:
            return {
                "error": "video-classification failed",
                "reason": "failed to persist uploaded video",
            }
        video_path = saved_path
    else:
        ap = spec.get("payload", {}).get("video_path")
        video_path = ap if isinstance(ap, str) and ap else "video.mp4"

    try:
        pl: VideoClassificationPipeline = pipeline(
            "video-classification",
            model=spec["model_id"],
            device=device_arg(dev),
        )
        out = pl(video_path)  # video_path is a concrete str
        if isinstance(out, list):
            for o in out:
                if "score" in o:
                    o["score"] = float(o["score"])
        return safe_json(out)
    except Exception as e:
        if "requires the PyAv library" in repr(e):
            return {
                "skipped": True,
                "reason": "missing dependency: PyAV",
                "hint": "Install with: pip install av",
            }
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {
            "error": "video-classification failed",
            "reason": repr(e),
            "hint": "This pipeline may need decord/pyav (pip install av) and GPU-appropriate deps.",
        }
    finally:
        # Cleanup temp file
        if video_file is not None and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass
