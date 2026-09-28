"""API tests with a stub detector: no torch, no weights, no Ollama needed."""
import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.api import main as api
from src.config import get_settings
from src.inference.detector import Detection
from src.inference.pipeline import InspectionPipeline
from src.inference.rag import KnowledgeBase
from src.inference.vlm import MockVLM


class StubDetector:
    path = "stub"
    imgsz = 320

    def __init__(self, dets):
        self.dets = dets

    def predict(self, img, conf=0.25, iou=0.5):
        return [d for d in self.dets if d.conf >= conf]


def make_client(tmp_path, monkeypatch, dets):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("VLM_PROVIDER", "mock")
    monkeypatch.setenv("REPORT_MODE", "template")
    s = get_settings()
    api.state.pipeline = InspectionPipeline(
        s, detector=StubDetector(dets), vlm=MockVLM(), kb=KnowledgeBase(s.kb_dir, use_embeddings=False)
    )
    return TestClient(api.app)


def png_bytes():
    img = np.full((120, 160, 3), 60, np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_health_and_inspect_fail(tmp_path, monkeypatch):
    dets = [Detection("short", 3, 0.9, [10, 10, 40, 40]), Detection("spur", 4, 0.3, [50, 50, 70, 70])]
    with make_client(tmp_path, monkeypatch, dets) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["vlm_provider"] == "mock"
        r = c.post("/inspect", files={"file": ("a.png", png_bytes(), "image/png")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["verdict"] == "FAIL"
        assert body["summary"] == {"short": 1, "spur": 1}
        assert body["vlm"]["calls"] == 1  # only the 0.3-confidence detection goes to the VLM
        assert body["annotated_image_b64"]
        assert "Recommended action" in body["report"]
        m = c.get("/metrics").json()
        assert m["requests"] >= 1 and m["verdicts"]["FAIL"] >= 1
        assert (tmp_path / "logs" / "predictions.jsonl").exists()


def test_pass_review_and_errors(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, []) as c:
        assert c.post("/inspect", files={"file": ("a.png", png_bytes(), "image/png")}).json()["verdict"] == "PASS"
        bad = c.post("/inspect", files={"file": ("a.txt", b"not an image", "text/plain")})
        assert bad.status_code == 400
        empty = c.post("/inspect", files={"file": ("a.png", b"", "image/png")})
        assert empty.status_code == 400
    with make_client(tmp_path, monkeypatch, [Detection("spur", 4, 0.3, [5, 5, 30, 30])]) as c:
        r = c.post("/inspect?use_vlm=false&return_image=false", files={"file": ("a.png", png_bytes(), "image/png")}).json()
        assert r["verdict"] == "REVIEW" and r["vlm"]["calls"] == 0 and r["annotated_image_b64"] is None
        assert (tmp_path / "logs" / "review_queue").exists()  # REVIEW images kept for error analysis


def test_conf_query_validation_and_detect_endpoint(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, [Detection("short", 3, 0.4, [1, 1, 20, 20])]) as c:
        assert c.post("/inspect?conf=5", files={"file": ("a.png", png_bytes(), "image/png")}).status_code == 422
        d = c.post("/detect?conf=0.5", files={"file": ("a.png", png_bytes(), "image/png")}).json()
        assert d["detections"] == []
        d = c.post("/detect?conf=0.3", files={"file": ("a.png", png_bytes(), "image/png")}).json()
        assert len(d["detections"]) == 1
