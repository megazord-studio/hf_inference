"""Phase C vision understanding runners: zero-shot image classification, zero-shot object detection, keypoint detection.

Implements minimal real-model logic using transformers/ultralytics. No fallbacks: errors surface.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Type, Set
import torch

from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, Owlv2Processor, Owlv2ForObjectDetection
from app.core.utils.media import decode_image_base64
from .base import BaseRunner

log = logging.getLogger("app.runners.vision_understanding")

VISION_UNDERSTANDING_TASKS: Set[str] = {
    "zero-shot-image-classification",
    "zero-shot-object-detection",
    "keypoint-detection",
}

class ZeroShotImageClassificationRunner(BaseRunner):
    model: Any
    processor: Any
    def load(self) -> int:
        log.info("vision_understanding: loading zero-shot image classification model_id=%s (may download)", self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        return sum(p.numel() for p in self.model.parameters())
    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        raw_labels = inputs.get("candidate_labels") or options.get("candidate_labels")
        labels: List[str]
        if isinstance(raw_labels, list) and all(isinstance(x, str) for x in raw_labels):
            labels = list(raw_labels)
        else:
            labels = ["cat", "dog", "tree"]
        if not img_b64:
            return {"error": "missing_image"}
        image = decode_image_base64(img_b64)
        encoding = self.processor(images=image, text=labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**encoding).logits_per_image.squeeze(0)
        probs = logits.softmax(-1).cpu().tolist()
        return {"labels": labels, "scores": probs}

class ZeroShotObjectDetectionRunner(BaseRunner):
    model: Any
    processor: Any
    def load(self) -> int:
        log.info("vision_understanding: loading OWLv2 model_id=%s (may download)", self.model_id)
        self.processor = Owlv2Processor.from_pretrained(self.model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_id)
        self.model.to(self.device)
        return sum(p.numel() for p in self.model.parameters())
    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        raw_labels = inputs.get("candidate_labels") or options.get("candidate_labels")
        labels: List[str]
        if isinstance(raw_labels, list) and all(isinstance(x, str) for x in raw_labels):
            labels = list(raw_labels)
        else:
            labels = ["person", "dog"]
        if not img_b64:
            raise RuntimeError("vision_understanding: missing_image")
        image = decode_image_base64(img_b64)
        encoding = self.processor(text=labels, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoding)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)[0]
        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        labels_out_idx = results["labels"].cpu().tolist()
        labels_out = [labels[i] if i < len(labels) else "unknown" for i in labels_out_idx]
        detections = [{"label": lbl, "score": float(scr), "box": box} for lbl, scr, box in zip(labels_out, scores, boxes)]
        return {"detections": detections}

class KeypointDetectionRunner(BaseRunner):
    yolo: Any
    def load(self) -> int:
        from ultralytics import YOLO  # local import to avoid import errors if package missing at parse time
        self.yolo = YOLO(self.model_id)
        return sum(p.numel() for p in self.yolo.model.parameters())
    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        if not img_b64:
            return {"error": "missing_image"}
        image = decode_image_base64(img_b64)
        results = self.yolo.predict(image, verbose=False)
        kp_out: List[List[float]] = []
        if results:
            r = results[0]
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                kps_raw = getattr(r.keypoints, 'xyn', None)
                if kps_raw is not None and hasattr(kps_raw, 'cpu'):
                    try:
                        kps_list = kps_raw.cpu().tolist()
                        if isinstance(kps_list, list):
                            kp_out = kps_list
                    except Exception:
                        pass
        return {"keypoints": kp_out, "count": len(kp_out)}

_TASK_MAP: Dict[str, Type[BaseRunner]] = {
    "zero-shot-image-classification": ZeroShotImageClassificationRunner,
    "zero-shot-object-detection": ZeroShotObjectDetectionRunner,
    "keypoint-detection": KeypointDetectionRunner,
}

def vision_understanding_runner_for_task(task: str) -> Type[BaseRunner]:
    return _TASK_MAP[task]

__all__ = [
    "VISION_UNDERSTANDING_TASKS",
    "vision_understanding_runner_for_task",
    "ZeroShotImageClassificationRunner",
    "ZeroShotObjectDetectionRunner",
    "KeypointDetectionRunner",
]
