"""Run the full pipeline on one image from the command line (no API needed).

    python scripts/run_inference.py path/to/board.jpg --out annotated.jpg
"""
import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from src.inference.pipeline import InspectionPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="annotated.jpg")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--report-mode", choices=["llm", "template"], default=None)
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    res = InspectionPipeline().run(img, use_vlm=not a.no_vlm, report_mode=a.report_mode)
    Path(a.out).write_bytes(base64.b64decode(res.pop("annotated_image_b64")))
    print(json.dumps(res, indent=2))
    print("annotated image ->", a.out)


if __name__ == "__main__":
    main()
