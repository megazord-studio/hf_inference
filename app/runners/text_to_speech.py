from transformers import pipeline
from app.helpers import device_arg, ensure_audio_path, save_wav, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, is_no_weight_files_error, soft_skip, soft_hint_error

def run_tts(spec, dev: str):
    try:
        pl = pipeline("text-to-speech", model=spec["model_id"], device=device_arg(dev))
        out = pl(spec["payload"]["tts_text"])
        audio = out["audio"] if isinstance(out, dict) and "audio" in out else out
        sr = out.get("sampling_rate", 16000) if isinstance(out, dict) else 16000
        path = ensure_audio_path(f"{spec['model_id'].replace('/','_')}_tts.wav")
        save_wav(audio, sr, path)
        safe_print_output({"audio_path": path, "sampling_rate": sr})
    except Exception as e:
        if is_gated_repo_error(e):
            soft_skip("gated model (no access/auth)", "Use facebook/mms-tts-* or request access."); return
        if is_missing_model_error(e) or is_no_weight_files_error(e):
            soft_skip("model not loadable (missing/unsupported weights)",
                      "Use facebook/mms-tts-* or microsoft/speecht5_tts (with embeddings)."); return
        if "speaker_embeddings" in repr(e):
            soft_skip("missing required speaker embeddings",
                      "Download xvectors (Matthijs/cmu-arctic-xvectors) and pass `speaker_embeddings`."); return
        soft_hint_error("text-to-speech failed", repr(e),
                        "Some models need external speaker embeddings (e.g., SpeechT5) or aren't supported by pipeline.")
