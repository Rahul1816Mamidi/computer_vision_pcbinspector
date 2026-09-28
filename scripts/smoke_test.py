"""5-minute end-to-end dry run on SYNTHETIC data (no Kaggle dataset, no Ollama needed).

    python scripts/smoke_test.py            # CPU is fine
    python scripts/smoke_test.py --device 0 # if you have a GPU

Steps: generate fake PCBs -> convert to YOLO -> train YOLOv8n for a few epochs -> export ONNX
       -> run the full pipeline (mock VLM) on one image. The tiny model will be weak; the point
       is to prove every stage runs, before you spend Kaggle GPU hours.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print("\n>>", " ".join(str(a) for a in args))
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()

    work = ROOT / "smoke_tmp"
    run("-m", "src.data.synthetic", "--out", work / "raw", "--n-per-class", 14)
    run("-m", "src.data.prepare_dataset", "--src", work / "raw", "--out", work / "dataset")
    run("-m", "src.training.train", "--data", work / "dataset" / "data.yaml", "--model", "yolov8n.pt",
        "--epochs", a.epochs, "--imgsz", 320, "--batch", 8, "--device", a.device, "--workers", 0,
        "--name", "smoke", "--project", work / "runs", "--models-dir", work / "models")
    run("-m", "src.optimization.export_onnx", "--weights", work / "models" / "best.pt")

    os.environ.update({"MODEL_PATH": str(work / "models" / "best.onnx"), "VLM_PROVIDER": "mock",
                       "REPORT_MODE": "template", "USE_EMBEDDINGS": "false", "CONF": "0.05"})
    sys.path.insert(0, str(ROOT))
    import cv2

    from src.inference.pipeline import InspectionPipeline

    img_path = next((work / "dataset" / "images" / "test").iterdir())
    res = InspectionPipeline().run(cv2.imread(str(img_path)), return_image=False)
    print("\nPipeline result for", img_path.name)
    print("verdict:", res["verdict"], "| detections:", res["summary"], "| timings:", res["timings_ms"])
    print(res["report"])
    print("\nSMOKE TEST PASSED: every stage ran. Now run the real thing on Kaggle.")


if __name__ == "__main__":
    main()
