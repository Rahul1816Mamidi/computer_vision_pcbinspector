"""Drawing helpers. Uses roboflow/supervision annotators, with a plain-OpenCV fallback."""
from __future__ import annotations

import base64
import cv2
import numpy as np


def annotate(image_bgr: np.ndarray, dets: list) -> np.ndarray:
    """Draw boxes + labels. `dets` are src.inference.detector.Detection objects."""
    img = image_bgr.copy()
    if not dets:
        return img
    h, w = img.shape[:2]
    scale = max(h, w) / 1000.0
    thickness = max(2, int(round(2 * scale)))
    labels = [f"{d.label} {d.conf:.2f}{' ?' if d.needs_review else ''}" for d in dets]
    try:
        import supervision as sv

        sv_det = sv.Detections(
            xyxy=np.array([d.xyxy for d in dets], dtype=float),
            confidence=np.array([d.conf for d in dets], dtype=float),
            class_id=np.array([d.cls_id for d in dets], dtype=int),
        )
        img = sv.BoxAnnotator(thickness=thickness).annotate(scene=img, detections=sv_det)
        img = sv.LabelAnnotator(text_scale=max(0.4, 0.5 * scale), text_thickness=max(1, thickness - 1),
                                text_padding=4).annotate(scene=img, detections=sv_det, labels=labels)
        return img
    except Exception:
        for d, text in zip(dets, labels):
            x1, y1, x2, y2 = [int(v) for v in d.xyxy]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness)
            cv2.putText(img, text, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.4, 0.5 * scale), (0, 0, 255), max(1, thickness - 1))
        return img


def to_b64_jpeg(img_bgr: np.ndarray, max_side: int = 1600, quality: int = 85) -> str:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
