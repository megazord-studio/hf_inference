from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Protocol
from typing import cast

import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor
from transformers import pipeline
from transformers.pipelines import ImageToTextPipeline
from transformers.pipelines import VisualQuestionAnsweringPipeline

from .helpers import device_arg
from .helpers import device_str
from .helpers import safe_print_output
from .types import RunnerSpec

# ---------- Protocol definitions for type hints ----------


class ModelProtocol(Protocol):
    """Protocol for transformer models."""

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Return model parameters."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the device the model is on."""
        ...

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """Generate output tokens."""
        ...


class ProcessorProtocol(Protocol):
    """Protocol for transformer processors."""

    def batch_decode(
        self, sequences: torch.Tensor, skip_special_tokens: bool = False
    ) -> list[str]:
        """Decode token sequences to strings."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Process inputs."""
        ...


# ---------- error detectors & friendly outputs ----------


def is_cuda_oom(e: Exception) -> bool:
    msg = repr(e).lower()
    return "cuda out of memory" in msg or "cuda oom" in msg


def is_missing_model_error(e: Exception) -> bool:
    msg = repr(e)
    return (
        "is not a local folder and is not a valid model identifier listed on"
        in msg
    )


def is_no_weight_files_error(e: Exception) -> bool:
    msg = repr(e)
    return (
        "does not appear to have a file named pytorch_model.bin" in msg
        or "model.safetensors" in msg
    )


def is_gated_repo_error(e: Exception) -> bool:
    msg = repr(e).lower()
    return (
        ("gated repo" in msg)
        or ("401 client error" in msg)
        or ("access to model" in msg and "restricted" in msg)
    )


def soft_skip(reason: str, hint: Optional[str] = None) -> None:
    out = {"skipped": True, "reason": reason}
    if hint:
        out["hint"] = hint
    safe_print_output(out)


def soft_hint_error(
    title: str, reason: str, hint: Optional[str] = None
) -> None:
    out = {"error": title, "reason": reason}
    if hint:
        out["hint"] = hint
    safe_print_output(out)


# ---------- VLM helpers ----------


def _cast_inputs_to_model_dtype(
    model: ModelProtocol, inputs: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        model_dtype = next(
            (p.dtype for p in model.parameters() if p is not None),
            torch.float16,
        )
    except StopIteration:
        model_dtype = torch.float16
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(model.device)
            out[k] = (
                v
                if v.dtype
                in (
                    torch.long,
                    torch.int,
                    torch.int32,
                    torch.int64,
                    torch.bool,
                )
                else v.to(model_dtype)
            )
        else:
            out[k] = v
    return out


def _decode_generate(
    model: ModelProtocol, processor: ProcessorProtocol, **inputs: Any
) -> str:
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=64)
    try:
        return processor.batch_decode(gen, skip_special_tokens=True)[0]
    except Exception:
        tok = getattr(processor, "tokenizer", None)
        return (
            tok.batch_decode(gen, skip_special_tokens=True)[0]
            if tok is not None
            else ""
        )


def _proc_inputs(
    processor: ProcessorProtocol,
    text: str,
    img: Image.Image,
    model: ModelProtocol,
) -> Dict[str, Any]:
    inputs = processor(text=text, images=img, return_tensors="pt")
    inputs = {
        k: (v.to(model.device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }
    return _cast_inputs_to_model_dtype(model, inputs)


def _final_caption_fallback(img: Image.Image, dev: str) -> Dict[str, Any]:
    try:
        # Build kwargs dict to avoid MyPy picking the wrong overload for pipeline(...)
        pl_kwargs: Dict[str, Any] = {
            "task": "image-to-text",
            "model": "nlpconnect/vit-gpt2-image-captioning",
            "device": device_arg(dev),
        }
        pl = cast(ImageToTextPipeline, pipeline(**pl_kwargs))
        # Call using positional 'inputs' to satisfy overload (inputs: Image | str)
        out = pl(img)
        if (
            isinstance(out, list)
            and out
            and isinstance(out[0], dict)
            and "generated_text" in out[0]
        ):
            return {"text": out[0]["generated_text"]}
        if isinstance(out, str):
            return {"text": out}
    except Exception as e:
        return {
            "error": "image-text-to-text failed",
            "reason": repr(e),
            "traceback": [],
        }
    return {
        "error": "image-text-to-text failed",
        "reason": "No compatible loader worked.",
        "traceback": [],
    }


def _vlm_minicpm(
    spec: RunnerSpec, img: Image.Image, prompt: str, dev: str
) -> Dict[str, Any]:
    try:
        proc = AutoProcessor.from_pretrained(
            spec["model_id"], trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            spec["model_id"], trust_remote_code=True, torch_dtype=torch.float16
        ).to(torch.device(device_str()))
        text = (
            prompt
            or "Caption this image in one sentence and include one color."
        )
        inputs = _proc_inputs(proc, text, img, model)
        txt = _decode_generate(model, proc, **inputs)
        return {"text": txt}
    except Exception:
        return _final_caption_fallback(img, dev)


def _vlm_llava(
    spec: RunnerSpec, img: Image.Image, prompt: str, dev: str
) -> Dict[str, Any]:
    try:
        q = prompt or "Describe this image in one sentence."
        # Build kwargs via dict to avoid overload confusion
        vqa_kwargs: Dict[str, Any] = {
            "task": "visual-question-answering",
            "model": spec["model_id"],
            "trust_remote_code": True,
            "device": device_arg(dev),
        }
        vqa = cast(VisualQuestionAnsweringPipeline, pipeline(**vqa_kwargs))
        ans = vqa(image=img, question=q)
        if (
            isinstance(ans, list)
            and ans
            and isinstance(ans[0], dict)
            and "answer" in ans[0]
        ):
            return {"text": ans[0]["answer"]}
        if isinstance(ans, dict) and "answer" in ans:
            return {"text": ans["answer"]}
        return {"text": str(ans)}
    except Exception:
        return _final_caption_fallback(img, dev)


def _vlm_florence2(
    spec: RunnerSpec, img: Image.Image, prompt: str, dev: str
) -> Dict[str, Any]:
    try:
        proc = AutoProcessor.from_pretrained(
            spec["model_id"], trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            spec["model_id"], trust_remote_code=True, torch_dtype=torch.float16
        ).to(torch.device(device_str()))  # type: ignore
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
    "is_cuda_oom",
    "is_missing_model_error",
    "is_no_weight_files_error",
    "is_gated_repo_error",
    "soft_skip",
    "soft_hint_error",
    "_final_caption_fallback",
    "_vlm_minicpm",
    "_vlm_llava",
    "_vlm_florence2",
    "_proc_inputs",
    "_decode_generate",
    "_cast_inputs_to_model_dtype",
]
