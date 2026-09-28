"""Central configuration.

Every value can be overridden with an environment variable (see .env.example).
Settings are read lazily via get_settings() so tests can change the environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Canonical class order used everywhere (dataset conversion, training, reports).
CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- detector -----------------------------------------------------------
    model_path: str
    device: str
    conf: float
    iou: float
    # --- VLM second opinion --------------------------------------------------
    vlm_provider: str  # ollama | gemini | mock | none
    ollama_host: str
    ollama_model: str
    gemini_api_key: str
    gemini_model: str
    vlm_timeout: float
    vlm_gate_conf: float  # detections below this confidence go to the VLM
    vlm_max_calls: int
    vlm_policy: str  # flag | apply
    # --- report / RAG --------------------------------------------------------
    report_mode: str  # llm | template
    report_timeout: float
    kb_dir: str
    embed_model: str
    use_embeddings: bool
    # --- serving -------------------------------------------------------------
    log_dir: str
    log_review_images: bool
    max_upload_mb: int


def get_settings() -> Settings:
    return Settings(
        model_path=os.getenv("MODEL_PATH", str(ROOT / "models" / "best.onnx")),
        device=os.getenv("DEVICE", "cpu"),
        conf=float(os.getenv("CONF", "0.25")),
        iou=float(os.getenv("IOU", "0.5")),
        vlm_provider=os.getenv("VLM_PROVIDER", "ollama").lower(),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        vlm_timeout=float(os.getenv("VLM_TIMEOUT", "90")),
        vlm_gate_conf=float(os.getenv("VLM_GATE_CONF", "0.5")),
        vlm_max_calls=int(os.getenv("VLM_MAX_CALLS", "3")),
        vlm_policy=os.getenv("VLM_POLICY", "flag").lower(),
        report_mode=os.getenv("REPORT_MODE", "llm").lower(),
        report_timeout=float(os.getenv("REPORT_TIMEOUT", "120")),
        kb_dir=os.getenv("KB_DIR", str(ROOT / "knowledge_base")),
        embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        use_embeddings=_bool("USE_EMBEDDINGS", True),
        log_dir=os.getenv("LOG_DIR", str(ROOT / "logs")),
        log_review_images=_bool("LOG_REVIEW_IMAGES", True),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "15")),
    )
