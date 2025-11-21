"""Phase D multimodal runner: ImageTextToText (VQA / caption QA).

Simplified baseline: use a vision-language model (placeholder: BLIP) to answer a question about an image.
No env gating; always-on discovery; errors surface.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Set, Type
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, pipeline
from app.core.utils.media import decode_image_base64
from .base import BaseRunner

log = logging.getLogger("app.runners.multimodal")

MULTIMODAL_TASKS: Set[str] = {"image-text-to-text"}

class ImageTextToTextRunner(BaseRunner):
    def load(self) -> int:
        mid = self.model_id.lower()
        # architecture classification
        if 'blip' in mid:
            self._arch = 'blip'
        elif 'llava' in mid:
            self._arch = 'llava'
        elif 'qwen' in mid:
            self._arch = 'qwen'
        elif 'paligemma' in mid:
            self._arch = 'paligemma'
        elif 'idefics' in mid:
            self._arch = 'idefics'
        elif 'minicpm' in mid:
            self._arch = 'minicpm'
        elif 'phi' in mid:
            self._arch = 'phi'
        else:
            self._arch = 'generic'
        if self._arch == 'blip':
            self.processor = BlipProcessor.from_pretrained(self.model_id)
            self.model = BlipForQuestionAnswering.from_pretrained(self.model_id)
            self.model.to(self.device)
            return sum(p.numel() for p in self.model.parameters())
        # shared pipeline for other architectures (VQA style)
        self.pipe = pipeline(
            task='visual-question-answering',
            model=self.model_id,
            device=0 if self.device and self.device.type=='cuda' else None,
            trust_remote_code=True,
        )
        model_ref = getattr(self.pipe, 'model', None)
        params = sum(p.numel() for p in model_ref.parameters()) if model_ref else 0
        return params
    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        img_b64 = inputs.get('image_base64')
        question = inputs.get('text') or options.get('question') or 'What is shown?'
        if not img_b64:
            return {'error': 'missing_image'}
        image = decode_image_base64(img_b64)
        if self._arch == 'blip':
            enc = self.processor(image, question, return_tensors='pt').to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_length=int(options.get('max_length', 32)))
            answer = self.processor.decode(out[0], skip_special_tokens=True)
            return {'answer': answer, 'arch': self._arch}
        out = self.pipe(image=image, question=question, max_new_tokens=int(options.get('max_length', 32)))
        first = out[0] if isinstance(out, list) else out
        answer = first.get('generated_text') or first.get('answer') or ''
        return {'answer': answer, 'arch': self._arch}
    def predict_stream(self, inputs: Dict[str, Any], options: Dict[str, Any]):
        if self._arch != 'blip':
            yield {'event': 'error', 'data': 'streaming_not_supported'}; return
        img_b64 = inputs.get('image_base64')
        question = inputs.get('text') or options.get('question') or 'What is shown?'
        if not img_b64:
            yield {'event': 'error', 'data': 'missing_image'}; return
        image = decode_image_base64(img_b64)
        enc = self.processor(image, question, return_tensors='pt').to(self.device)
        max_len = int(options.get('max_length', 32))
        with torch.no_grad():
            seq = self.model.generate(**enc, max_length=max_len)
        text = self.processor.decode(seq[0], skip_special_tokens=True)
        # naive chunk stream
        for i in range(0, len(text), 8):
            yield {'event': 'token', 'data': text[i:i+8]}
        yield {'event': 'done', 'data': text}

_TASK_MAP = {"image-text-to-text": ImageTextToTextRunner}

def multimodal_runner_for_task(task: str) -> Type[BaseRunner]:
    return _TASK_MAP[task]

__all__ = ["MULTIMODAL_TASKS", "multimodal_runner_for_task", "ImageTextToTextRunner"]
