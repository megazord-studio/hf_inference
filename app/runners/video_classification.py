from transformers import pipeline
from app.helpers import device_arg, safe_json, get_upload_file_path
import os
from app.utilities import is_gated_repo_error, is_missing_model_error

def run_video_classification(spec, dev: str):
    """
    Run video classification inference.
    Accepts either video_path or UploadFile from spec["files"]["video"].
    Returns the result as a dictionary instead of printing.
    """
    # Handle UploadFile or fallback to path
    video_file = spec.get("files", {}).get("video")
    if video_file is not None:
        # Save temporarily for pipeline
        temp_path = f"/tmp/video_{os.getpid()}.mp4"
        video_path = get_upload_file_path(video_file, temp_path)
    else:
        video_path = spec["payload"].get("video_path", "video.mp4")
    
    try:
        pl = pipeline("video-classification", model=spec["model_id"], device=device_arg(dev))
        out = pl(video_path)
        if isinstance(out, list):
            for o in out:
                if "score" in o:
                    o["score"] = float(o["score"])
        return safe_json(out)
    except Exception as e:
        if "requires the PyAv library" in repr(e):
            return {"skipped": True, "reason": "missing dependency: PyAV", "hint": "Install with: pip install av"}
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {"skipped": True, "reason": "model not found on Hugging Face"}
        return {"error": "video-classification failed", "reason": repr(e),
                "hint": "This pipeline may need decord/pyav (pip install av) and GPU-appropriate deps."}
    finally:
        # Cleanup temp file
        if video_file is not None and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass
