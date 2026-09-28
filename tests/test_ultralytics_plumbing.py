"""Exercises Detector / train / export / evaluate / benchmark with a FAKE `ultralytics` module.

This verifies OUR code (argument plumbing, file handling, metrics/report writing, ONNX-vs-pt path logic).
It cannot verify Ultralytics itself: that is what scripts/smoke_test.py is for.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.data.prepare_dataset import prepare
from src.data.synthetic import generate

CALLS = {}


class _T:  # tiny tensor stand-in: supports .cpu().numpy()
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


class _Boxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy, self.cls, self.conf = _T(xyxy), _T(cls), _T(conf)

    def __len__(self):
        return len(self.cls)


class FakeYOLO:
    def __init__(self, path, task=None):
        self.path, self.task = str(path), task
        self.names = {0: "missing_hole", 1: "mouse_bite", 2: "open_circuit", 3: "short", 4: "spur", 5: "spurious_copper"}

    def predict(self, img, imgsz=None, conf=None, iou=None, device=None, verbose=None):
        CALLS["predict"] = dict(imgsz=imgsz, conf=conf, iou=iou, device=device)
        return [types.SimpleNamespace(boxes=_Boxes([[5, 5, 30, 30], [40, 40, 80, 80]], [3, 4], [0.4, 0.9]))]

    def train(self, **kw):
        CALLS["train"] = kw
        best = Path(kw["project"]) / kw["name"] / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"weights")
        self.trainer = types.SimpleNamespace(best=str(best))
        return types.SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.61, "metrics/mAP50-95(B)": 0.33, "fitness": 0.4})

    def export(self, **kw):
        CALLS["export"] = kw
        out = Path(self.path).with_suffix(".onnx")
        out.write_bytes(b"onnx")
        return str(out)

    def val(self, **kw):
        CALLS["val"] = kw
        box = types.SimpleNamespace(map50=0.7, map=0.4, mp=0.6, mr=0.5,
                                    ap50=np.array([0.9, 0.5]), ap_class_index=np.array([0, 3]))
        return types.SimpleNamespace(box=box)


@pytest.fixture(autouse=True)
def fake_ultralytics(monkeypatch, tmp_path):
    CALLS.clear()
    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    import src.utils.experiment_log as el
    monkeypatch.setattr(el, "LOG_PATH", tmp_path / "exp.jsonl")
    monkeypatch.setattr(el, "MD_PATH", tmp_path / "exp.md")


def test_detector_paths_meta_and_sorting(tmp_path, monkeypatch):
    from src.inference.detector import Detector, resolve_model_path

    (tmp_path / "best.onnx").write_bytes(b"x")
    (tmp_path / "model_meta.json").write_text(json.dumps({"imgsz": 1024}))
    d = Detector(str(tmp_path / "best.onnx"), device="cpu")
    assert d.imgsz == 1024 and d.path.suffix == ".onnx"
    out = d.predict(np.zeros((100, 100, 3), np.uint8), conf=0.3, iou=0.5)
    assert [x.label for x in out] == ["spur", "short"]  # sorted by confidence
    assert CALLS["predict"]["imgsz"] == 1024 and CALLS["predict"]["conf"] == 0.3
    assert Detector(str(tmp_path / "best.onnx"), imgsz=640).imgsz == 640  # explicit override wins
    import src.inference.detector as det_mod
    monkeypatch.setattr(det_mod, "ROOT", tmp_path / "empty_project")
    with pytest.raises(FileNotFoundError):
        resolve_model_path(str(tmp_path / "missing.pt"))


def test_train_copies_weights_and_writes_meta(tmp_path, monkeypatch):
    from src.training import train

    monkeypatch.setattr(sys, "argv", ["train", "--data", str(tmp_path / "data.yaml"), "--epochs", "2", "--imgsz", "512",
                                      "--device", "cpu", "--project", str(tmp_path / "runs"), "--name", "t",
                                      "--models-dir", str(tmp_path / "models")])
    train.main()
    assert (tmp_path / "models" / "best.pt").read_bytes() == b"weights"
    meta = json.loads((tmp_path / "models" / "model_meta.json").read_text())
    assert meta["imgsz"] == 512 and len(meta["classes"]) == 6
    k = CALLS["train"]
    assert k["imgsz"] == 512 and k["epochs"] == 2 and k["flipud"] == 0.5 and k["degrees"] == 0.0
    log = json.loads((tmp_path / "exp.jsonl").read_text().splitlines()[0])
    assert log["kind"] == "train" and log["metrics"]["mAP50"] == 0.61


def test_export_onnx_updates_meta(tmp_path, monkeypatch):
    from src.optimization import export_onnx

    (tmp_path / "best.pt").write_bytes(b"w")
    (tmp_path / "model_meta.json").write_text(json.dumps({"imgsz": 800}))
    monkeypatch.setattr(sys, "argv", ["export", "--weights", str(tmp_path / "best.pt")])
    export_onnx.main()
    assert (tmp_path / "best.onnx").exists()
    assert CALLS["export"]["imgsz"] == 800 and CALLS["export"]["format"] == "onnx" and CALLS["export"]["dynamic"] is False
    assert json.loads((tmp_path / "model_meta.json").read_text())["onnx"] is True


def test_evaluate_main_writes_reports(tmp_path, monkeypatch):
    from src.training import evaluate

    raw = generate(tmp_path / "raw", n_per_class=6, seed=5)
    ds = tmp_path / "ds"
    prepare(raw, ds, ("rotat",), seed=1)
    (tmp_path / "best.pt").write_bytes(b"w")
    out = tmp_path / "reports"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--data", str(ds / "data.yaml"), "--weights", str(tmp_path / "best.pt"),
                                      "--imgsz", "320", "--device", "cpu", "--out", str(out)])
    evaluate.main()
    m = json.loads((out / "metrics_test.json").read_text())
    assert m["mAP50"] == 0.7 and m["per_class_AP50"] == {"missing_hole": 0.9, "short": 0.5}
    assert CALLS["val"]["split"] == "test" and CALLS["val"]["imgsz"] == 320
    assert (out / "error_analysis.json").exists() and "Confusion matrix" in (out / "evaluation_report.md").read_text()


def test_benchmark_main(tmp_path, monkeypatch):
    import cv2

    from src.optimization import benchmark

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(3):
        cv2.imwrite(str(imgs / f"{i}.jpg"), np.zeros((64, 64, 3), np.uint8))
    (tmp_path / "a.pt").write_bytes(b"1")
    (tmp_path / "b.onnx").write_bytes(b"2")
    monkeypatch.setattr(sys, "argv", ["bench", "--models", str(tmp_path / "a.pt"), str(tmp_path / "b.onnx"),
                                      "--images", str(imgs), "--n", "3", "--warmup", "1", "--out", str(tmp_path / "r")])
    benchmark.main()
    rows = json.loads((tmp_path / "r" / "latency_benchmark.json").read_text())
    assert [r["model"] for r in rows] == ["a.pt", "b.onnx"] and rows[0]["speedup_vs_first"] == 1.0
