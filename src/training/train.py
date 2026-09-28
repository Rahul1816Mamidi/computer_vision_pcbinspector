"""Fine-tune a YOLO detector on the prepared PCB dataset (run this on a Kaggle GPU).

    python -m src.training.train --data /kaggle/working/dataset/data.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from src.config import CLASS_NAMES, ROOT
from src.utils.experiment_log import log_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", default="yolov8s.pt", help="pretrained checkpoint (auto-downloaded)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1024, help="PCB defects are tiny; 1024 beats 640 clearly")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0", help="GPU index, or 'cpu'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="pcb_yolov8s")
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    a = ap.parse_args()

    from ultralytics import YOLO  # lazy import

    model = YOLO(a.model)
    t0 = time.time()
    results = model.train(
        data=str(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
        workers=a.workers, device=a.device, seed=a.seed, project=a.project, name=a.name, exist_ok=True,
        # PCBs have no canonical orientation: flips are safe, big rotations are not needed
        flipud=0.5, fliplr=0.5, degrees=0.0, mosaic=1.0, close_mosaic=10, scale=0.4,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, cos_lr=True, plots=True,
    )
    minutes = (time.time() - t0) / 60

    best = Path(model.trainer.best)
    a.models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, a.models_dir / "best.pt")
    meta = {
        "imgsz": a.imgsz, "classes": CLASS_NAMES, "base_model": a.model, "epochs_requested": a.epochs,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "onnx": False,
    }
    (a.models_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))

    rd = getattr(results, "results_dict", {}) or {}
    metrics = {k.replace("metrics/", "").replace("(B)", ""): float(v) for k, v in rd.items() if "metrics/" in k}
    metrics["train_minutes"] = round(minutes, 1)
    log_experiment("train", a.name, {"model": a.model, "epochs": a.epochs, "imgsz": a.imgsz, "batch": a.batch}, metrics)
    print(f"\nBest weights copied to {a.models_dir / 'best.pt'}")
    print("Validation metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
