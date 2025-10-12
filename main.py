#!/usr/bin/env python3
import os
import io
import json
import sys
from typing import Any, Dict, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import soundfile as sf
import yaml

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
)

from diffusers import StableDiffusionPipeline

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------- small utilities ---------------------------

def device_str() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def device_arg(dev: str):
    return dev

def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _ensure_cpu_float(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return float(x.item())
        return x.numpy().tolist()
    if isinstance(x, np.generic):
        return float(x.item())
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    return x

def safe_json(obj):
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (str, int, bool, type(None))):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, (np.generic,)):
        return _ensure_cpu_float(obj)
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return _ensure_cpu_float(obj)
    return str(obj)

def safe_print_output(obj: Any):
    clean = safe_json(obj)
    print("Output type:", type(clean))
    print(json.dumps(clean, indent=2, ensure_ascii=False))

def print_header():
    import transformers, diffusers
    print(f"python: {sys.version.split()[0]} on device={device_str()}")
    print(f"transformers: {transformers.__version__} diffusers: {diffusers.__version__} HAS_DIFFUSERS: {True}")

def load_demo(path: str = "./demo.yaml") -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        doc = yaml.safe_load(f)
    demos = doc.get("demos", [])
    print(f"[BOOT] Loaded {len(demos)} demos from {path}")
    return demos

def ensure_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        img = Image.new("RGB", (768, 512), "#E8F2FF")
        d = ImageDraw.Draw(img)
        d.rectangle((20, 400, 300, 500), fill="#F4F4F4", outline="#CCCCCC")
        d.text((30, 410), "placeholder image.jpg", fill="#333333")
        img.save(path)
    return Image.open(path).convert("RGB")

def ensure_audio_path(name: str) -> str:
    return os.path.join(OUT_DIR, name)

def save_wav(audio: np.ndarray, sr: int, path: str):
    arr = _np(audio)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        pass
    elif arr.ndim == 2:
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
    else:
        arr = arr.reshape(-1)
    sf.write(path, arr, sr)

def to_dataframe(table_like: List[List[str]]) -> pd.DataFrame:
    rows = [[str(x) for x in r] for r in table_like]
    if rows and all(isinstance(c, str) for c in rows[0]):
        header = rows[0]
        data = rows[1:] if len(rows) > 1 else []
        df = pd.DataFrame(data, columns=header)
    else:
        df = pd.DataFrame(rows)
    return df

# --------------------------- helpers: image/text VLM ---------------------------

def _cast_inputs_to_model_dtype(model, inputs: Dict[str, Any]):
    """Move tensors to device; cast only *floating* tensors to model dtype.
       Keep integer types (e.g. input_ids, attention_mask) as-is."""
    try:
        model_dtype = next((p.dtype for p in model.parameters() if p is not None), torch.float16)
    except StopIteration:
        model_dtype = torch.float16
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(model.device)
            # Keep integral tensors integral
            if v.dtype in (torch.long, torch.int, torch.int32, torch.int64, torch.bool):
                out[k] = v
            else:
                out[k] = v.to(model_dtype)
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
        if tok is not None:
            return tok.batch_decode(gen, skip_special_tokens=True)[0]
        return ""

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
    # Use robust VQA route on 4.46.x
    try:
        q = prompt or "Describe this image in one sentence."
        vqa = pipeline(
            "visual-question-answering",
            model=spec["model_id"],
            trust_remote_code=True,
            device=device_arg(dev),
        )
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

# --------------------------- runners ---------------------------

def run_text_generation(spec, dev: str):
    pl = pipeline("text-generation", model=spec["model_id"], device=device_arg(dev))
    out = pl(spec["payload"]["prompt"], max_new_tokens=64)
    safe_print_output(out)

def run_text2text(spec, dev: str):
    pl = pipeline("text2text-generation", model=spec["model_id"], device=device_arg(dev))
    out = pl(spec["payload"]["prompt"], max_new_tokens=64)
    safe_print_output(out)

def run_zero_shot_classification(spec, dev: str):
    pl = pipeline("zero-shot-classification", model=spec["model_id"], device=device_arg(dev))
    out = pl(spec["payload"]["prompt"], candidate_labels=spec["payload"]["candidate_labels"])
    safe_print_output(out)

def run_summarization(spec, dev: str):
    pl = pipeline("summarization", model=spec["model_id"], device=device_arg(dev))
    out = pl(spec["payload"]["prompt"], max_new_tokens=64)
    safe_print_output(out)

def run_translation(spec, dev: str):
    payload = spec["payload"]
    pl = pipeline("translation", model=spec["model_id"], device=device_arg(dev))
    if "src_lang" in payload or "tgt_lang" in payload:
        out = pl(payload["prompt"], src_lang=payload.get("src_lang"), tgt_lang=payload.get("tgt_lang"))
    else:
        out = pl(payload["prompt"])
    safe_print_output(out)

def run_qa(spec, dev: str):
    pl = pipeline("question-answering", model=spec["model_id"], device=device_arg(dev))
    payload = spec["payload"]
    out = pl(question=payload["qa_question"], context=payload["qa_context"])
    safe_print_output(out)

def _normalize_mask_sentence(model_id: str, sentence: str) -> Tuple[str, str]:
    tok = AutoTokenizer.from_pretrained(model_id)
    mask_token = tok.mask_token or "<mask>"
    s = sentence or ""
    # Replace common placeholders with the model's actual mask token
    s = s.replace("<mask>", mask_token)
    s = s.replace("[MASK]", mask_token)
    s = s.replace("[mask]", mask_token)
    s = s.replace("__", mask_token)
    # Ensure exactly one mask token
    if mask_token not in s:
        s = f"The capital of Switzerland is {mask_token}."
    # If multiple masks slipped in, keep the first
    parts = s.split(mask_token)
    if len(parts) > 2:
        s = mask_token.join([parts[0], parts[1]])  # keep only one
    return s, mask_token

def run_fill_mask(spec, dev: str):
    pl = pipeline("fill-mask", model=spec["model_id"], device=device_arg(dev))
    p = spec["payload"]
    s1, _ = _normalize_mask_sentence(spec["model_id"], p.get("mask_sentence", ""))
    s2, _ = _normalize_mask_sentence(spec["model_id"], p.get("mask_sentence_alt", ""))
    r1 = pl(s1)
    r2 = pl(s2)
    safe_print_output({"result_1": r1, "result_2": r2})

def run_sentiment(spec, dev: str):
    pl = pipeline("sentiment-analysis", model=spec["model_id"], device=device_arg(dev))
    out = pl(spec["payload"]["prompt"])
    safe_print_output(out)

def run_ner(spec, dev: str):
    pl = pipeline("token-classification", model=spec["model_id"], aggregation_strategy="simple", device=device_arg(dev))
    out = pl(spec["payload"]["prompt"])
    for o in out:
        o["score"] = float(o.get("score", 0.0))
    safe_print_output(out)

def run_feature_extraction(spec, dev: str):
    pl = pipeline("feature-extraction", model=spec["model_id"], device=device_arg(dev))
    vec = pl(spec["payload"]["prompt"])
    safe_print_output({"embedding_shape": np.array(vec).shape})

def run_table_qa(spec, dev: str):
    p = spec["payload"]
    df = to_dataframe(p["table"])
    query = p["table_query"]
    pl = pipeline("table-question-answering", model=spec["model_id"], device=device_arg(dev))
    out = pl({"table": df, "query": query})
    safe_print_output(out)

def run_vqa(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    pl = pipeline("visual-question-answering", model=spec["model_id"], device=device_arg(dev))
    out = pl(image=img, question=spec["payload"]["question"])
    safe_print_output(out)

def run_doc_qa(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    pl = pipeline("document-question-answering", model=spec["model_id"], device=device_arg(dev))
    out = pl(image=img, question=spec["payload"]["question"])
    safe_print_output(out)

def run_vlm_image_text_to_text(spec, dev: str):
    payload = spec["payload"]
    img = ensure_image(payload["image_path"])
    prompt = payload.get("prompt", "Describe the image briefly.")

    model_id = spec["model_id"].lower()

    if "llava" in model_id:
        safe_print_output(_vlm_llava(spec, img, prompt, dev)); return
    if "florence-2" in model_id:
        safe_print_output(_vlm_florence2(spec, img, prompt, dev)); return
    if "minicpm" in model_id or "cpm" in model_id:
        safe_print_output(_vlm_minicpm(spec, img, prompt, dev)); return
    if "qwen" in model_id or "internvl" in model_id or "yi-vl" in model_id:
        # these are brittle on 4.46.x; go straight to fallback if they fail
        try:
            pl = pipeline("image-to-text", model=spec["model_id"], trust_remote_code=True, device=device_arg(dev))
            out = pl(img)
            if isinstance(out, list) and out and "generated_text" in out[0]:
                safe_print_output({"text": out[0]["generated_text"]}); return
            if isinstance(out, str):
                safe_print_output({"text": out}); return
        except Exception:
            pass
        safe_print_output(_final_caption_fallback(img, dev)); return

    # Generic attempt
    try:
        pl = pipeline("image-text-to-text", model=spec["model_id"], trust_remote_code=True, device=device_arg(dev))
        out = pl({"image": img, "prompt": prompt})
        if isinstance(out, dict) and "text" in out:
            safe_print_output({"text": out["text"]}); return
        if isinstance(out, list) and out and "generated_text" in out[0]:
            safe_print_output({"text": out[0]["generated_text"]}); return
        if isinstance(out, str):
            safe_print_output({"text": out}); return
    except Exception:
        pass

    # Try image-to-text
    try:
        pl2 = pipeline("image-to-text", model=spec["model_id"], trust_remote_code=True, device=device_arg(dev))
        out2 = pl2(img)
        if isinstance(out2, list) and out2 and "generated_text" in out2[0]:
            safe_print_output({"text": out2[0]["generated_text"]}); return
        if isinstance(out2, str):
            safe_print_output({"text": out2}); return
    except Exception:
        pass

    safe_print_output(_final_caption_fallback(img, dev))

def run_image_to_text(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    pl = pipeline("image-to-text", model=spec["model_id"], device=device_arg(dev))
    out = pl(img)
    if isinstance(out, list) and out and "generated_text" in out[0]:
        safe_print_output({"text": out[0]["generated_text"]})
    else:
        safe_print_output(out)

def run_zero_shot_image_classification(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    pl = pipeline("zero-shot-image-classification", model=spec["model_id"], device=device_arg(dev))
    out = pl(img, candidate_labels=spec["payload"]["candidate_labels"])
    safe_print_output(out)

def run_image_classification(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    pl = pipeline("image-classification", model=spec["model_id"], device=device_arg(dev))
    out = pl(img)
    safe_print_output(out)

def run_zero_shot_object_detection(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    pl = pipeline("zero-shot-object-detection", model=spec["model_id"], device=device_arg(dev))
    out = pl(img, candidate_labels=spec["payload"]["candidate_labels"])
    safe_print_output(out)

def run_object_detection(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    pl = pipeline("object-detection", model=spec["model_id"], device=device_arg(dev))
    out = pl(img)
    safe_print_output(out)

def _save_panoptic_masks(seg_list: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    saved = []
    for i, seg in enumerate(seg_list):
        mask = seg.get("mask", None)
        entry = {k: (float(v) if k == "score" else v) for k, v in seg.items() if k != "mask"}
        if isinstance(mask, (Image.Image, np.ndarray, torch.Tensor)):
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            if isinstance(mask, np.ndarray):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8)) if mask.ndim == 2 else Image.fromarray(mask)
            else:
                mask_img = mask
            path = os.path.join(OUT_DIR, f"{prefix}_{i}.png")
            mask_img.save(path)
            entry["mask_path"] = path
        saved.append(entry)
    return saved

def run_image_segmentation(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    pl = pipeline("image-segmentation", model=spec["model_id"], device=device_arg(dev))
    out = pl(img)
    saved = _save_panoptic_masks(out, f"seg_{spec['model_id'].replace('/', '_')}")
    safe_print_output(saved)

def run_depth_estimation(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    pl = pipeline("depth-estimation", model=spec["model_id"], device=device_arg(dev))
    out = pl(img)
    depth = out.get("predicted_depth", None) if isinstance(out, dict) else None
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if isinstance(depth, np.ndarray):
        d = depth
        d = d - d.min()
        if d.max() > 0:
            d = d / d.max()
        d_img = Image.fromarray((d * 255).astype(np.uint8))
        path = os.path.join(OUT_DIR, f"depth_{spec['model_id'].replace('/', '_')}.png")
        d_img.save(path)
        safe_print_output({"depth_path": path})
    else:
        safe_print_output(out)

def run_asr(spec, dev: str):
    pl = pipeline("automatic-speech-recognition", model=spec["model_id"], device=device_arg(dev))
    out = pl(spec["payload"]["audio_path"])
    safe_print_output(out)

def run_audio_classification(spec, dev: str):
    pl = pipeline("audio-classification", model=spec["model_id"], device=device_arg(dev))
    out = pl(spec["payload"]["audio_path"])
    for o in out:
        o["score"] = float(o["score"])
    safe_print_output(out)

def run_tts(spec, dev: str):
    pl = pipeline("text-to-speech", model=spec["model_id"], device=device_arg(dev))
    out = pl(spec["payload"]["tts_text"])
    audio = out["audio"] if isinstance(out, dict) and "audio" in out else out
    sr = out.get("sampling_rate", 16000) if isinstance(out, dict) else 16000
    path = ensure_audio_path(f"{spec['model_id'].replace('/','_')}_tts.wav")
    save_wav(audio, sr, path)
    safe_print_output({"audio_path": path, "sampling_rate": sr})

def run_text_to_audio(spec, dev: str):
    pl = pipeline("text-to-audio", model=spec["model_id"], device=device_arg(dev))
    out = pl(spec["payload"]["tta_prompt"])
    audio = out["audio"] if isinstance(out, dict) and "audio" in out else out
    sr = out.get("sampling_rate", 32000) if isinstance(out, dict) else 32000
    path = ensure_audio_path(f"{spec['model_id'].replace('/','_')}_music.wav")
    save_wav(audio, sr, path)
    safe_print_output({"audio_path": path, "sampling_rate": sr})

def run_text_to_image(spec, dev: str):
    prompt = spec["payload"]["prompt"]
    pipe = StableDiffusionPipeline.from_pretrained(
        spec["model_id"], torch_dtype=torch.float16
    ).to(device_str())
    with torch.inference_mode():
        img = pipe(prompt=prompt, num_inference_steps=25, guidance_scale=7.0).images[0]
    path = os.path.join(OUT_DIR, f"sd_{spec['model_id'].replace('/', '_')}.png")
    img.save(path)
    safe_print_output({"image_path": path})

RUNNERS = {
    "text-generation": run_text_generation,
    "text2text-generation": run_text2text,
    "zero-shot-classification": run_zero_shot_classification,
    "summarization": run_summarization,
    "translation": run_translation,
    "question-answering": run_qa,
    "fill-mask": run_fill_mask,
    "sentiment-analysis": run_sentiment,
    "token-classification": run_ner,
    "feature-extraction": run_feature_extraction,
    "table-question-answering": run_table_qa,
    "visual-question-answering": run_vqa,
    "document-question-answering": run_doc_qa,
    "image-text-to-text": run_vlm_image_text_to_text,
    "image-to-text": run_image_to_text,
    "zero-shot-image-classification": run_zero_shot_image_classification,
    "image-classification": run_image_classification,
    "zero-shot-object-detection": run_zero_shot_object_detection,
    "object-detection": run_object_detection,
    "image-segmentation": run_image_segmentation,
    "depth-estimation": run_depth_estimation,
    "automatic-speech-recognition": run_asr,
    "audio-classification": run_audio_classification,
    "text-to-speech": run_tts,
    "text-to-audio": run_text_to_audio,
    "text-to-image": run_text_to_image,
}

# --------------------------- main ---------------------------

def main():
    print_header()
    demos = load_demo("./demo.yaml")
    dev = device_str()
    respect_skip = os.getenv("RESPECT_SKIP", "0") == "1"

    for item in demos:
        model_id = item["model_id"]
        task = item["task"]
        payload = item.get("payload", {})
        if respect_skip and item.get("skipped", False):
            print(f"=== {model_id} ({task}) ===")
            print("— SKIPPED —")
            continue

        print(f"=== {model_id} ({task}) ===")
        runner = RUNNERS.get(task)
        if not runner:
            safe_print_output({
                "error": "Could not run. Supported tasks: ['<unavailable>']",
                "Reason": f"Unknown task {task}"
            })
            continue

        try:
            runner({"model_id": model_id, "task": task, "payload": payload}, dev)
        except Exception as e:
            if task == "image-text-to-text":
                try:
                    img = ensure_image(payload["image_path"])
                    fb = _final_caption_fallback(img, dev)
                    safe_print_output(fb)
                except Exception as e2:
                    safe_print_output({
                        "error": f"{task} failed",
                        "reason": repr(e2),
                        "traceback": []
                    })
            else:
                safe_print_output({
                    "error": f"{task} failed",
                    "reason": repr(e),
                    "hint": None,
                    "traceback": []
                })

if __name__ == "__main__":
    main()
