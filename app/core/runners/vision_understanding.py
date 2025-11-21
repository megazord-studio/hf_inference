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
    def load(self) -> int:
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        return sum(p.numel() for p in self.model.parameters())
    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        labels: List[str] = inputs.get("candidate_labels") or options.get("candidate_labels") or ["cat", "dog", "tree"]
        if not img_b64:
            return {"error": "missing_image"}
        image = decode_image_base64(img_b64)
        encoding = self.processor(images=image, text=labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**encoding).logits_per_image.squeeze(0)
        probs = logits.softmax(-1).cpu().tolist()
        return {"labels": labels, "scores": probs}

class ZeroShotObjectDetectionRunner(BaseRunner):
    def load(self) -> int:
        self.processor = Owlv2Processor.from_pretrained(self.model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_id)
        self.model.to(self.device)
        return sum(p.numel() for p in self.model.parameters())
    def predict(self, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        img_b64 = inputs.get("image_base64")
        labels: List[str] = inputs.get("candidate_labels") or options.get("candidate_labels") or ["person", "dog"]
        if not img_b64:
            return {"error": "missing_image"}
        image = decode_image_base64(img_b64)
        encoding = self.processor(text=labels, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoding)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]
        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        labels_out = [labels[i] if i < len(labels) else "unknown" for i in results["labels"].cpu().tolist()]
        detections = [
            {"label": lbl, "score": float(scr), "box": box}
            for lbl, scr, box in zip(labels_out, scores, boxes)
        ]
        return {"detections": detections}

class KeypointDetectionRunner(BaseRunner):
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
        # Extract keypoints from first result
        kp_out = []
        if results:
            r = results[0]
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                kps = r.keypoints.xyn.cpu().tolist()  # normalized
                kp_out = kps
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
