"""Lightweight production monitoring: in-memory metrics + JSONL prediction log."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests = 0
        self.errors = 0
        self.verdicts: Counter = Counter()
        self.labels: Counter = Counter()
        self.vlm_calls = 0
        self.latency = deque(maxlen=500)
        self.confidences = deque(maxlen=2000)

    def record(self, result: Dict[str, Any]) -> None:
        with self._lock:
            self.requests += 1
            self.verdicts[result["verdict"]] += 1
            for d in result["detections"]:
                self.labels[d["label"]] += 1
                self.confidences.append(d["confidence"])
            self.vlm_calls += result["vlm"]["calls"]
            self.latency.append(result["timings_ms"]["total_ms"])

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lat = np.array(self.latency) if self.latency else np.array([0.0])
            conf = np.array(self.confidences) if self.confidences else np.array([0.0])
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "requests": self.requests,
                "errors": self.errors,
                "verdicts": dict(self.verdicts),
                "detections_by_label": dict(self.labels),
                "vlm_calls": self.vlm_calls,
                "latency_ms": {"mean": round(float(lat.mean()), 1), "p50": round(float(np.percentile(lat, 50)), 1),
                               "p95": round(float(np.percentile(lat, 95)), 1)},
                "confidence": {"mean": round(float(conf.mean()), 3), "p10": round(float(np.percentile(conf, 10)), 3)},
            }


class PredictionLogger:
    """Appends one JSON line per request; optionally stores REVIEW images for later error analysis."""

    def __init__(self, log_dir: str, save_review_images: bool = True) -> None:
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.review_dir = self.dir / "review_queue"
        self.save_review_images = save_review_images
        self._lock = threading.Lock()

    def log(self, request_id: str, raw_bytes: bytes, image_bgr: np.ndarray, result: Dict[str, Any]) -> None:
        digest = hashlib.sha1(raw_bytes).hexdigest()[:12]
        rec = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": request_id,
            "image_sha1": digest,
            "image_size": result["image_size"],
            "verdict": result["verdict"],
            "n_detections": len(result["detections"]),
            "labels": result["summary"],
            "max_conf": max([d["confidence"] for d in result["detections"]], default=None),
            "vlm_calls": result["vlm"]["calls"],
            "report_mode": result["report_mode_used"],
            "timings_ms": result["timings_ms"],
        }
        with self._lock, open(self.dir / "predictions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if self.save_review_images and result["verdict"] == "REVIEW":
            self.review_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.review_dir / f"{digest}.jpg"), image_bgr)
