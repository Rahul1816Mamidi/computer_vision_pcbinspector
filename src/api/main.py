"""FastAPI service.  Run:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.api.monitoring import Metrics, PredictionLogger
from src.config import get_settings
from src.inference.pipeline import InspectionPipeline


class DetectionOut(BaseModel):
    label: str
    confidence: float
    box_xyxy: List[float]
    size_px: List[float]
    needs_review: bool
    vlm: Optional[Dict[str, Any]] = None


class InspectResponse(BaseModel):
    request_id: str
    verdict: Literal["PASS", "REVIEW", "FAIL"]
    summary: Dict[str, int]
    detections: List[DetectionOut]
    report: str
    report_mode_used: str
    retrieved_context: List[Dict[str, Any]]
    image_size: List[int]
    vlm: Dict[str, Any]
    timings_ms: Dict[str, float]
    annotated_image_b64: Optional[str] = None


class State:
    pipeline: Optional[InspectionPipeline] = None
    error: Optional[str] = None
    metrics = Metrics()
    logger: Optional[PredictionLogger] = None


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    if state.pipeline is None:  # tests inject their own pipeline
        try:
            state.pipeline = InspectionPipeline(s)
        except FileNotFoundError as e:  # keep the server up so /health can explain what is missing
            state.error = str(e)
    state.logger = PredictionLogger(s.log_dir, s.log_review_images)
    yield


app = FastAPI(
    title="PCB Defect Inspector",
    version="1.0.0",
    description="YOLO defect detection + VLM second opinion + RAG-grounded inspection report.",
    lifespan=lifespan,
)


def _decode(raw: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Upload a JPEG/PNG/BMP file.")
    return img


async def _read_upload(file: UploadFile) -> bytes:
    limit = get_settings().max_upload_mb * 1024 * 1024
    raw = await file.read()
    if len(raw) > limit:
        raise HTTPException(status_code=413, detail=f"File larger than {get_settings().max_upload_mb} MB")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    return raw


def _require_pipeline() -> None:
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail=state.error or "pipeline not loaded")


@app.get("/health")
def health() -> Dict[str, Any]:
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail=state.error or "pipeline not loaded")
    return {"status": "ok", **state.pipeline.status()}


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    return state.metrics.snapshot()


@app.post("/inspect", response_model=InspectResponse)
async def inspect(
    file: UploadFile = File(...),
    conf: Optional[float] = Query(None, ge=0.01, le=0.99, description="detector confidence threshold"),
    use_vlm: bool = Query(True, description="ask the VLM about uncertain detections"),
    report_mode: Optional[Literal["llm", "template"]] = Query(None),
    return_image: bool = Query(True, description="include annotated image as base64 JPEG"),
):
    raw = await _read_upload(file)
    img = _decode(raw)
    _require_pipeline()
    request_id = uuid.uuid4().hex[:12]
    try:
        # sync, CPU-bound work: run it off the event loop
        result = await run_in_threadpool(
            state.pipeline.run, img, conf, use_vlm, report_mode, return_image
        )
    except Exception as e:
        state.metrics.record_error()
        raise HTTPException(status_code=500, detail=f"inference failed: {type(e).__name__}: {e}")
    state.metrics.record(result)
    if state.logger:
        state.logger.log(request_id, raw, img, result)
    return {"request_id": request_id, **result}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: Optional[float] = Query(None, ge=0.01, le=0.99),
):
    """Detector only (no VLM, no report): the fast path."""
    raw = await _read_upload(file)
    img = _decode(raw)
    _require_pipeline()
    result = await run_in_threadpool(state.pipeline.run, img, conf, False, "template", False)
    return {"detections": result["detections"], "verdict": result["verdict"],
            "detect_ms": result["timings_ms"]["detect_ms"]}
