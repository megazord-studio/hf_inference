from transformers import pipeline
from app.helpers import device_arg, ensure_audio_path, save_wav, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, is_no_weight_files_error, soft_skip

def run_text_to_audio(spec, dev: str):
    try:
        pl = pipeline("text-to-audio", model=spec["model_id"], device=device_arg(dev))
        out = pl(spec["payload"]["tta_prompt"])
        audio = out["audio"] if isinstance(out, dict) and "audio" in out else out
        sr = out.get("sampling_rate", 32000) if isinstance(out, dict) else 32000
        path = ensure_audio_path(f"{spec['model_id'].replace('/','_')}_music.wav")
        save_wav(audio, sr, path)
        safe_print_output({"audio_path": path, "sampling_rate": sr})
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)", "Try facebook/musicgen-*"); return
        if is_missing_model_error(e) or is_no_weight_files_error(e):
            soft_skip("model not loadable (missing/unsupported weights)", "Try facebook/musicgen-*"); return
        safe_print_output({"error": "text-to-audio failed", "reason": repr(e)})
