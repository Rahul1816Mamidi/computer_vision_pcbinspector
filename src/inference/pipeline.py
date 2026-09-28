"""End-to-end inspection pipeline: detect -> (VLM second opinion) -> RAG -> report."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import Settings, get_settings
from src.inference.detector import Detection, Detector
from src.inference.rag import KnowledgeBase
from src.inference.report import decide_verdict, generate_report, summarize
from src.inference.vlm import BaseVLM, build_vlm, make_crop, verify_crop
from src.utils.viz import annotate, to_b64_jpeg


class InspectionPipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        detector: Optional[Any] = None,
        vlm: Optional[BaseVLM] = None,
        kb: Optional[KnowledgeBase] = None,
    ):
        self.s = settings or get_settings()
        self.detector = detector or Detector(self.s.model_path, device=self.s.device)
        self.vlm = vlm if vlm is not None else build_vlm(self.s)
        self.kb = kb if kb is not None else KnowledgeBase(self.s.kb_dir, self.s.embed_model, self.s.use_embeddings)
        self._lock = threading.Lock()  # the underlying model is not guaranteed thread-safe

    # -- health -------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "model": str(getattr(self.detector, "path", "custom")),
            "imgsz": getattr(self.detector, "imgsz", None),
            "vlm_provider": self.s.vlm_provider,
            "vlm_ready": bool(self.vlm and self.vlm.available()),
            "vlm_model": self.s.ollama_model if self.s.vlm_provider == "ollama" else self.s.gemini_model
            if self.s.vlm_provider == "gemini" else None,
            "rag_backend": self.kb.backend if self.kb else None,
            "vlm_policy": self.s.vlm_policy,
        }

    # -- VLM second opinion --------------------------------------------------
    def _second_opinion(self, image: np.ndarray, dets: List[Detection]):
        """Ask the VLM about uncertain detections. Returns (possibly filtered detections, number of VLM calls)."""
        gate, max_calls = self.s.vlm_gate_conf, self.s.vlm_max_calls
        uncertain = [d for d in dets if d.conf < gate][:max_calls]  # dets are sorted by confidence
        rejected: List[Detection] = []
        for d in uncertain:
            v = verify_crop(self.vlm, make_crop(image, d.xyxy), d.label)
            d.vlm = v
            if v["verdict"] in {"reject", "relabel"}:
                d.needs_review = True
                if self.s.vlm_policy == "apply":
                    if v["verdict"] == "reject":
                        rejected.append(d)
                    elif v["label"]:
                        d.extra["detector_label"] = d.label
                        d.label = v["label"]
        return [d for d in dets if d not in rejected], len(uncertain)

    # -- main entry ------------------------------------------------------------
    def run(
        self,
        image_bgr: np.ndarray,
        conf: Optional[float] = None,
        use_vlm: bool = True,
        report_mode: Optional[str] = None,
        return_image: bool = True,
    ) -> Dict[str, Any]:
        t: Dict[str, float] = {}
        t0 = time.perf_counter()
        with self._lock:
            dets = self.detector.predict(image_bgr, conf=conf or self.s.conf, iou=self.s.iou)
        t["detect_ms"] = (time.perf_counter() - t0) * 1000

        vlm_calls = 0
        t1 = time.perf_counter()
        if use_vlm and self.vlm is not None and dets and self.vlm.available():
            dets, vlm_calls = self._second_opinion(image_bgr, dets)
        t["vlm_ms"] = (time.perf_counter() - t1) * 1000

        verdict = decide_verdict(dets, self.s.vlm_gate_conf)

        t2 = time.perf_counter()
        mode = (report_mode or self.s.report_mode).lower()
        report, mode_used, contexts = generate_report(mode, verdict, dets, self.kb, self.vlm)
        t["report_ms"] = (time.perf_counter() - t2) * 1000
        t["total_ms"] = (time.perf_counter() - t0) * 1000

        h, w = image_bgr.shape[:2]
        return {
            "verdict": verdict,
            "summary": summarize(dets),
            "detections": [d.to_dict() for d in dets],
            "report": report,
            "report_mode_used": mode_used,
            "retrieved_context": [{"doc": c["doc"], "section": c["section"], "score": c.get("score")} for c in contexts],
            "image_size": [w, h],
            "vlm": {"provider": self.s.vlm_provider, "calls": vlm_calls, "policy": self.s.vlm_policy},
            "timings_ms": {k: round(v, 1) for k, v in t.items()},
            "annotated_image_b64": to_b64_jpeg(annotate(image_bgr, dets)) if return_image else None,
        }
