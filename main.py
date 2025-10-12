#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import time
import traceback
from typing import Any, List, Optional

import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.tapas")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pydantic import BaseModel

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

# ---------------------------
# Pretty printing helpers
# ---------------------------

def _safe_obj(o: Any) -> Any:
    try:
        import torch as _torch
    except Exception:
        _torch = None

    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return f"np.ndarray(shape={list(o.shape)}, dtype={str(o.dtype)})"
    if _torch is not None and isinstance(o, _torch.Tensor):
        return f"torch.Tensor(shape={list(o.shape)}, dtype={str(o.dtype)})"
    if isinstance(o, Image.Image):
        return f"PIL.Image(size={o.size}, mode={o.mode})"
    if isinstance(o, (list, tuple)):
        return [_safe_obj(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _safe_obj(v) for k, v in o.items()}
    return str(o)


def shorten(o: Any, max_len: int = 800) -> str:
    try:
        safe = _safe_obj(o)
        s = json.dumps(safe, ensure_ascii=False)
    except Exception:
        s = str(o)
    return s if len(s) <= max_len else s[:max_len] + "..."


# ---------------------------
# Pydantic types
# ---------------------------

class DemoPayload(BaseModel):
    prompt: Optional[str] = None
    qa_question: Optional[str] = None
    qa_context: Optional[str] = None
    mask_sentence: Optional[str] = None
    mask_sentence_alt: Optional[str] = None
    table: Optional[List[List[str]]] = None
    table_query: Optional[str] = None
    image_path: Optional[str] = None
    audio_path: Optional[str] = None

    model_config = {'arbitrary_types_allowed': True}


class DemoSpec(BaseModel):
    model_id: str
    task: str
    payload: DemoPayload


# ---------------------------
# Utilities
# ---------------------------

def load_pil_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


# ---------------------------
# VLM (Qwen2.5-VL) runner
# ---------------------------

def vlm_qwen_it2t(model_id: str, image: Image.Image, prompt: str, device: str) -> str:
    if not prompt:
        raise RuntimeError("VLM demo requires a non-empty prompt in payload.prompt.")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    templated = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=[templated], images=[image], return_tensors="pt")

    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            v = v.to(model.device)
            if v.dtype in (torch.float32, torch.bfloat16):
                v = v.half()
            inputs[k] = v

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=64)

    out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return out


# ---------------------------
# Inference router
# ---------------------------

def build_pipeline(task: str, model_id: str, device: str) -> Pipeline:
    dev = 0 if device.startswith("cuda") and torch.cuda.is_available() else -1
    return pipeline(task=task, model=model_id, device=dev)


def unified_infer(spec: DemoSpec, device: str):
    t0 = time.time()
    mid, task, p = spec.model_id, spec.task, spec.payload

    # Vision-Language
    if task == "image-text-to-text":
        if not p.image_path:
            raise RuntimeError("VLM demo requires payload.image_path.")
        img = load_pil_image(p.image_path)
        out = vlm_qwen_it2t(mid, img, p.prompt or "", device)
        return out, time.time() - t0

    # All other tasks -> pipelines
    pl = build_pipeline(task, mid, device)

    # Text-only
    if task == "text-generation":
        out = pl(p.prompt)
    elif task == "text2text-generation":
        out = pl(p.prompt)
    elif task == "summarization":
        out = pl(p.prompt)
    elif task == "translation":
        if "m2m100" in mid.lower():
            out = pl(p.prompt, src_lang="en", tgt_lang="de")
        else:
            out = pl(p.prompt)
    elif task == "question-answering":
        out = pl({"question": p.qa_question, "context": p.qa_context})
    elif task == "sentiment-analysis":
        out = pl(p.prompt)
    elif task == "token-classification":
        out = pl(p.prompt)
    elif task == "feature-extraction":
        out = pl(p.prompt)

    # Fill-mask
    elif task == "fill-mask":
        tok = AutoTokenizer.from_pretrained(mid)
        mask_tok = tok.mask_token or "[MASK]"
        sent = p.mask_sentence if mask_tok == "[MASK]" else (p.mask_sentence_alt or p.mask_sentence)
        if sent is None or mask_tok not in sent:
            raise RuntimeError(f"Mask sentence must contain the model's mask token ({mask_tok}).")
        out = pl(sent)

    # Table QA
    elif task == "table-question-answering":
        df = pd.DataFrame(p.table or [])
        if len(df.columns):
            df.columns = [str(c) for c in df.columns]
        df = df.astype(str)
        out = pl(table=df, query=p.table_query or "")

    # Vision-only
    elif task in {"image-to-text", "image-classification", "object-detection", "image-segmentation", "depth-estimation"}:
        if not p.image_path:
            raise RuntimeError(f"{task} demo requires payload.image_path.")
        img = load_pil_image(p.image_path)
        out = pl(img)

    # Audio
    elif task == "automatic-speech-recognition":
        if not p.audio_path or not os.path.exists(p.audio_path):
            raise RuntimeError("ASR demo requires payload.audio_path (existing file).")
        out = pl(p.audio_path)
    elif task == "audio-classification":
        if not p.audio_path or not os.path.exists(p.audio_path):
            raise RuntimeError("Audio-classification demo requires payload.audio_path (existing file).")
        out = pl(p.audio_path)

    else:
        raise ValueError(f"Unsupported task: {task}")

    return out, time.time() - t0


# ---------------------------
# Spec loading (YAML only; no auto-mapping)
# ---------------------------

def load_specs_from_yaml(yaml_path: str) -> List[DemoSpec]:
    import yaml
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"{yaml_path} not found")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    demos = data.get("demos", [])
    if not isinstance(demos, list) or not demos:
        raise ValueError("demo.yaml is present but contains no 'demos' list.")

    specs: List[DemoSpec] = []
    for item in demos:
        model_id = item["model_id"]
        task = item["task"]
        payload_dict = item.get("payload", {}) or {}
        specs.append(DemoSpec(model_id=model_id, task=task, payload=DemoPayload(**payload_dict)))
    return specs


# ---------------------------
# Main
# ---------------------------

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"python: {torch.__version__} on device={device}")
    print("transformers:", __import__("transformers").__version__, "diffusers:", "unavailable", "HAS_DIFFUSERS:", False)

    YAML_PATH = "./demo.yaml"
    specs = load_specs_from_yaml(YAML_PATH)
    print(f"[BOOT] Loaded {len(specs)} demos from {YAML_PATH}")

    for spec in specs:
        mid, task = spec.model_id, spec.task
        print(f"=== {mid} ({task}) ===")
        try:
            out, took = unified_infer(spec, device)
            print(f"[{mid}] took {took:.2f}s")
            print("Output type:", type(out))
            print(shorten(out))
        except Exception as e:
            try:
                supported = list(getattr(pipeline, "task_mapping").keys())
            except Exception:
                supported = []
            err = "".join(traceback.format_exception_only(type(e), e)).strip()
            tb = "".join(traceback.format_tb(e.__traceback__))
            shown_supported = supported if supported else ["<unavailable>"]
            print("Could not run. Supported tasks:", shown_supported)
            print("Reason:", err)
            print("Traceback (last calls):")
            print(tb)


if __name__ == "__main__":
    main()
