from transformers import pipeline
from app.helpers import device_arg, safe_json, get_upload_file_path
import os
from app.utilities import is_gated_repo_error, is_missing_model_error, is_no_weight_files_error

def run_asr(spec, dev: str):
    """
    Run automatic speech recognition inference.
    Accepts either audio_path or UploadFile from spec["files"]["audio"].
    Returns the result as a dictionary instead of printing.
    """
    # Handle UploadFile or fallback to path
    audio_file = spec.get("files", {}).get("audio")
    if audio_file is not None:
        # Save temporarily for pipeline
        temp_path = f"/tmp/audio_{os.getpid()}.wav"
        audio_path = get_upload_file_path(audio_file, temp_path)
    else:
        audio_path = spec["payload"].get("audio_path", "audio.wav")
    
    try:
        pl = pipeline("automatic-speech-recognition", model=spec["model_id"], device=device_arg(dev))
        out = pl(audio_path)
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e) or is_no_weight_files_error(e):
            return {"skipped": True, "reason": "model not loadable (missing files)"}
        return {"error": "automatic-speech-recognition failed", "reason": repr(e)}
    finally:
        # Cleanup temp file
        if audio_file is not None and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
