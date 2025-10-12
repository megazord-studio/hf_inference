from typing import Any, Dict, Tuple
import torch
from transformers import pipeline, AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
from PIL import Image

from .helpers import device_arg, device_str, safe_print_output

# ---------- error detectors & friendly outputs ----------

def is_cuda_oom(e: Exception) -> bool:
    msg = repr(e).lower()
    return "cuda out of memory" in msg or "cuda oom" in msg

def is_missing_model_error(e: Exception) -> bool:
    msg = repr(e)
    return "is not a local folder and is not a valid model identifier listed on" in msg

def is_no_weight_files_error(e: Exception) -> bool:
    msg = repr(e)
    return "does not appear to have a file named pytorch_model.bin" in msg or "model.safetensors" in msg

def is_gated_repo_error(e: Exception) -> bool:
    msg = repr(e).lower()
    return ("gated repo" in msg) or ("401 client error" in msg) or ("access to model" in msg and "restricted" in msg)

def soft_skip(reason: str, hint: str = None):
    out = {"skipped": True, "reason": reason}
    if hint:
        out["hint"] = hint
    safe_print_output(out)

def soft_hint_error(title: str, reason: str, hint: str = None):
    out = {"error": title, "reason": reason}
    if hint:
        out["hint"] = hint
    safe_print_output(out)

# ---------- VLM helpers ----------

def _cast_inputs_to_model_dtype(model, inputs: Dict[str, Any]):
    try:
        model_dtype = next((p.dtype for p in model.parameters() if p is not None), torch.float16)
    except StopIteration:
        model_dtype = torch.float16
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(model.device)
            out[k] = v if v.dtype in (torch.long, torch.int, torch.int32, torch.int64, torch.bool) else v.to(model_dtype)
        else:
            out[k] = v
    return out

def _decode_generate(model, processor, **inputs):
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=64)
    try:
        return processor.batch_decode(gen, skip_special_tokens=True)[0]
    except Exception:
        tok = getattr(processor, "tokenizer", None)
        return tok.batch_decode(gen, skip_special_tokens=True)[0] if tok is not None else ""

def _proc_inputs(processor, text: str, img: Image.Image, model):
    inputs = processor(text=text, images=img, return_tensors="pt")
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    return _cast_inputs_to_model_dtype(model, inputs)

def _final_caption_fallback(img: Image.Image, dev: str):
    try:
        pl = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=device_arg(dev))
        out = pl(img)
        if isinstance(out, list) and out and "generated_text" in out[0]:
            return {"text": out[0]["generated_text"]}
        if isinstance(out, str):
            return {"text": out}
    except Exception as e:
        return {"error": "image-text-to-text failed", "reason": repr(e), "traceback": []}
    return {"error": "image-text-to-text failed", "reason": "No compatible loader worked.", "traceback": []}

def _vlm_minicpm(spec, img: Image.Image, prompt: str, dev: str):
    try:
        proc = AutoProcessor.from_pretrained(spec["model_id"], trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            spec["model_id"], trust_remote_code=True, torch_dtype=torch.float16
        ).to(device_str())
        text = prompt or "Caption this image in one sentence and include one color."
        inputs = _proc_inputs(proc, text, img, model)
        txt = _decode_generate(model, proc, **inputs)
        return {"text": txt}
    except Exception:
        return _final_caption_fallback(img, dev)

def _vlm_llava(spec, img: Image.Image, prompt: str, dev: str):
    try:
        q = prompt or "Describe this image in one sentence."
        vqa = pipeline("visual-question-answering", model=spec["model_id"], trust_remote_code=True, device=device_arg(dev))
        ans = vqa(image=img, question=q)
        if isinstance(ans, list) and ans and isinstance(ans[0], dict) and "answer" in ans[0]:
            return {"text": ans[0]["answer"]}
        if isinstance(ans, dict) and "answer" in ans:
            return {"text": ans["answer"]}
        return {"text": str(ans)}
    except Exception:
        return _final_caption_fallback(img, dev)

def _vlm_florence2(spec, img: Image.Image, prompt: str, dev: str):
    try:
        proc = AutoProcessor.from_pretrained(spec["model_id"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec["model_id"], trust_remote_code=True, torch_dtype=torch.float16
        ).to(device_str())
        text = prompt or "Describe the image briefly and include one color."
        inputs = _proc_inputs(proc, text, img, model)
        txt = _decode_generate(model, proc, **inputs)
        if len(txt.strip()) < 6:
            text2 = "Give a concise one-sentence caption and explicitly mention a color."
            inputs2 = _proc_inputs(proc, text2, img, model)
            txt = _decode_generate(model, proc, **inputs2)
        return {"text": txt}
    except Exception:
        return _final_caption_fallback(img, dev)

# expose for runners
__all__ = [
    "is_cuda_oom", "is_missing_model_error", "is_no_weight_files_error", "is_gated_repo_error",
    "soft_skip", "soft_hint_error",
    "_final_caption_fallback", "_vlm_minicpm", "_vlm_llava", "_vlm_florence2",
    "_proc_inputs", "_decode_generate", "_cast_inputs_to_model_dtype",
]
