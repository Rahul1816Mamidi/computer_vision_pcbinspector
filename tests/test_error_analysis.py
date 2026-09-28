"""error_analysis() with a stub detector on a synthetic dataset (no torch needed)."""
from pathlib import Path

from src.config import CLASS_NAMES
from src.data.prepare_dataset import prepare
from src.data.synthetic import generate
from src.inference.detector import Detection
from src.training.evaluate import error_analysis, read_yolo_labels, write_markdown
from src.training.evaluate_pipeline import score


def test_error_analysis_counts(tmp_path: Path, monkeypatch):
    raw = generate(tmp_path / "raw", n_per_class=6, seed=3)
    ds = tmp_path / "ds"
    prepare(raw, ds, ("rotat",), seed=1)
    img_dir, lab_dir = ds / "images" / "test", ds / "labels" / "test"

    state = {"i": 0}
    files = sorted(img_dir.iterdir())

    class Det:
        def predict(self, img, conf=0.25, iou=0.5):
            f = files[state["i"]]
            state["i"] += 1
            h, w = img.shape[:2]
            boxes, cls = read_yolo_labels(lab_dir / f"{f.stem}.txt", w, h)
            out = [Detection(CLASS_NAMES[c], c, 0.9, b.tolist()) for b, c in zip(boxes, cls) if c != 0]  # drop class 0
            out.append(Detection("short", 3, 0.6, [0, 0, 5, 5]))  # hallucinated box in each image
            return out

    res = error_analysis(Det(), img_dir, lab_dir, 0.25, 0.5, tmp_path / "rep")
    n_missing_hole = res["per_class"]["missing_hole"]["fn"]
    assert n_missing_hole >= 1 and res["per_class"]["missing_hole"]["recall"] == 0.0
    assert res["per_class"]["spur"]["recall"] == 1.0
    assert res["per_class"]["short"]["fp"] == len(files)  # one junk box per image
    nC = len(CLASS_NAMES)
    assert res["confusion_matrix"]["matrix"][0][nC] == n_missing_hole  # missed -> background column
    assert (tmp_path / "rep" / "error_cases").exists() and res["worst_images"]
    write_markdown(res, {"mAP50": 0.5, "mAP50-95": 0.3, "precision": 0.4, "recall": 0.6}, tmp_path / "r.md")
    assert "Confusion matrix" in (tmp_path / "r.md").read_text()


def test_score_variants():
    import numpy as np

    gt = {"a.jpg": (np.array([[0, 0, 10, 10]]), [0])}
    good = [("a.jpg", [{"box": [0, 0, 10, 10], "cls": 0, "conf": 0.9}])]
    bad = [("a.jpg", [{"box": [0, 0, 10, 10], "cls": 0, "conf": 0.9}, {"box": [50, 50, 60, 60], "cls": 1, "conf": 0.3}])]
    assert score(good, gt, 0.5)["f1"] == 1.0
    s = score(bad, gt, 0.5)
    assert s["fp"] == 1 and s["precision"] == 0.5 and s["recall"] == 1.0
