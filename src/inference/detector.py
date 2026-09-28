"""YOLO detector wrapper. Loads .onnx (preferred, fast on CPU) or .pt weights.

ultralytics is imported lazily so unit tests / tooling that never run the
model do not need torch installed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import CLASS_NAMES, ROOT


@dataclass
class Detection:
    label: str
    cls_id: int
    conf: float
    xyxy: List[float]
    needs_review: bool = False
    vlm: Optional[Dict[str, Any]] = None  # VLM second-opinion result, if any
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        x1, y1, x2, y2 = self.xyxy
        return {
            "label": self.label,
            "confidence": round(float(self.conf), 4),
            "box_xyxy": [round(float(v), 1) for v in self.xyxy],
            "size_px": [round(float(x2 - x1), 1), round(float(y2 - y1), 1)],
            "needs_review": self.needs_review,
            "vlm": self.vlm,
        }


def resolve_model_path(path: Optional[str] = None) -> Path:
    """Use the given path; otherwise prefer models/best.onnx, then models/best.pt."""
    if path and Path(path).exists():
        return Path(path)
    for cand in (ROOT / "models" / "best.onnx", ROOT / "models" / "best.pt"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "No model weights found. Train on Kaggle (notebooks/kaggle_train.ipynb) and copy "
        "best.pt / best.onnx into ./models/ (see README)."
    )


def load_meta(model_dir: Path) -> Dict[str, Any]:
    f = model_dir / "model_meta.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    return {}


class Detector:
    def __init__(self, model_path: Optional[str] = None, imgsz: Optional[int] = None, device: str = "cpu"):
        from ultralytics import YOLO  # lazy import

        self.path = resolve_model_path(model_path)
        self.model = YOLO(str(self.path), task="detect")
        meta = load_meta(self.path.parent)
        self.imgsz = int(imgsz or meta.get("imgsz") or 640)
        self.device = device
        try:
            names = self.model.names  # for ONNX this is read from the model metadata
        except Exception:
            names = None
        if isinstance(names, dict) and names:
            self.names = {int(k): v for k, v in names.items()}
        else:
            self.names = dict(enumerate(CLASS_NAMES))

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.5) -> List[Detection]:
        res = self.model.predict(
            image_bgr, imgsz=self.imgsz, conf=conf, iou=iou, device=self.device, verbose=False
        )[0]
        out: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for b, c, s in zip(xyxy, cls, cf):
            out.append(Detection(label=self.names.get(int(c), str(c)), cls_id=int(c), conf=float(s), xyxy=b.tolist()))
        out.sort(key=lambda d: -d.conf)
        return out
