"""Export the trained YOLO weights to ONNX (and optionally int8-quantize them).

    python -m src.optimization.export_onnx --weights models/best.pt

ONNX Runtime is typically much faster than PyTorch on CPU, which is what the Docker
service runs on. The export is fixed-size (imgsz from model_meta.json) for the best speed.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.config import ROOT
from src.inference.detector import load_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--quantize", action="store_true",
                    help="EXPERIMENTAL dynamic int8 weights -> models/best_int8.onnx. "
                         "Always re-check mAP (src.training.evaluate) before using it.")
    a = ap.parse_args()

    from ultralytics import YOLO

    meta = load_meta(a.weights.parent)
    imgsz = a.imgsz or meta.get("imgsz") or 640
    exported = Path(YOLO(str(a.weights)).export(format="onnx", imgsz=imgsz, opset=a.opset,
                                                simplify=True, dynamic=False))
    target = a.weights.parent / "best.onnx"
    if exported.resolve() != target.resolve():
        shutil.copy2(exported, target)
    print("ONNX model:", target, f"({target.stat().st_size / 1e6:.1f} MB)")

    if a.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = a.weights.parent / "best_int8.onnx"
        quantize_dynamic(str(target), str(q), weight_type=QuantType.QUInt8)
        print("Quantized model:", q, f"({q.stat().st_size / 1e6:.1f} MB)")

    meta.update({"imgsz": imgsz, "onnx": True, "opset": a.opset})
    (a.weights.parent / "model_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
