"""Architecture detection for multimodal models."""
from __future__ import annotations


def detect_arch(model_id: str) -> str:
    """Detect model architecture from model_id."""
    mid = model_id.lower()

    if "blip" in mid:
        return "blip"
    if "llava" in mid:
        return "llava"
    if any(k in mid for k in ["qwen-vl", "qwen/vl", "qwen-vl-chat"]):
        return "qwen_vl"
    if any(k in mid for k in ["minicpm", "minicpm-v", "minicpm_v", "minicpm-o", "minicpmv"]):
        return "minicpm_vlm"
    if "yi-vl" in mid or "yi/vl" in mid:
        return "yi_vl"
    if "internvl" in mid:
        return "internvl"
    if "kosmos-2" in mid or "kosmos2" in mid:
        return "kosmos2"
    if "florence-2" in mid or "florence2" in mid:
        return "florence2"
    if "cogvlm" in mid:
        return "cogvlm"
    if any(k in mid for k in ["idefics", "paligemma", "vl-", "vision-language", "gemma"]):
        return "vlm"
    return "generic_vqa"
