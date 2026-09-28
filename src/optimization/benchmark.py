"""Latency benchmark: PyTorch (.pt) vs ONNX Runtime (.onnx) on the same images.

    python -m src.optimization.benchmark --models models/best.pt models/best.onnx \
        --images dataset/images/test --n 30

Measures end-to-end predict() time (preprocess + inference + NMS), after warm-up.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.config import ROOT
from src.utils.experiment_log import log_experiment


def load_images(folder: Path, n: int) -> List[np.ndarray]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[:n]
    imgs = [cv2.imread(str(f)) for f in files]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        raise SystemExit(f"no images in {folder}")
    return imgs


def bench(model_path: str, imgs: List[np.ndarray], warmup: int, imgsz, device: str) -> Dict:
    from src.inference.detector import Detector

    det = Detector(model_path, imgsz=imgsz, device=device)
    for i in range(warmup):
        det.predict(imgs[i % len(imgs)], conf=0.25)
    times = []
    for img in imgs:
        t = time.perf_counter()
        det.predict(img, conf=0.25)
        times.append((time.perf_counter() - t) * 1000)
    a = np.array(times)
    return {"model": Path(model_path).name, "size_mb": round(Path(model_path).stat().st_size / 1e6, 1),
            "imgsz": det.imgsz, "device": device, "n": len(times),
            "mean_ms": round(float(a.mean()), 1), "p50_ms": round(float(np.percentile(a, 50)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1), "fps": round(1000 / max(float(a.mean()), 1e-6), 2),
            "_mean_raw": float(a.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    a = ap.parse_args()

    imgs = load_images(a.images, a.n)
    rows = [bench(m, imgs, a.warmup, a.imgsz, a.device) for m in a.models]
    base = rows[0]["_mean_raw"]
    for r in rows:
        r["speedup_vs_first"] = round(base / max(r.pop("_mean_raw"), 1e-6), 2)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "latency_benchmark.json").write_text(json.dumps(rows, indent=2))
    md = ["# Latency benchmark", "", f"Device: {a.device}. Images: {len(imgs)} from `{a.images}`.", "",
          "| model | size MB | imgsz | mean ms | p50 ms | p95 ms | FPS | speedup vs first |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['model']} | {r['size_mb']} | {r['imgsz']} | {r['mean_ms']} | {r['p50_ms']} | {r['p95_ms']} | {r['fps']} | {r['speedup_vs_first']}x |" for r in rows]
    (a.out / "latency_benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    for r in rows:
        log_experiment("benchmark", r["model"], {"device": r["device"], "imgsz": r["imgsz"]},
                       {"mean_ms": r["mean_ms"], "fps": r["fps"]})


if __name__ == "__main__":
    main()
