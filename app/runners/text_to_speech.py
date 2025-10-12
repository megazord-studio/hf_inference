from transformers import pipeline
from app.helpers import device_arg, audio_to_bytes, safe_json
from app.utilities import is_gated_repo_error, is_missing_model_error, is_no_weight_files_error

def run_tts(spec, dev: str):
    """
    Run text-to-speech inference.
    Returns audio as bytes in a dictionary with metadata.
    """
    try:
        pl = pipeline("text-to-speech", model=spec["model_id"], device=device_arg(dev))
        out = pl(spec["payload"]["tts_text"])
        audio = out["audio"] if isinstance(out, dict) and "audio" in out else out
        sr = out.get("sampling_rate", 16000) if isinstance(out, dict) else 16000
        
        # Convert audio to bytes
        audio_bytes = audio_to_bytes(audio, sr)
        return {
            "file_data": audio_bytes,
            "file_name": f"{spec['model_id'].replace('/', '_')}_tts.wav",
            "content_type": "audio/wav",
            "sampling_rate": sr
        }
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)", "hint": "Use facebook/mms-tts-* or request access."}
        if is_missing_model_error(e) or is_no_weight_files_error(e):
            return {"skipped": True, "reason": "model not loadable (missing/unsupported weights)",
                    "hint": "Use facebook/mms-tts-* or microsoft/speecht5_tts (with embeddings)."}
        if "speaker_embeddings" in repr(e):
            return {"skipped": True, "reason": "missing required speaker embeddings",
                    "hint": "Download xvectors (Matthijs/cmu-arctic-xvectors) and pass `speaker_embeddings`."}
        return {"error": "text-to-speech failed", "reason": repr(e),
                "hint": "Some models need external speaker embeddings (e.g., SpeechT5) or aren't supported by pipeline."}
