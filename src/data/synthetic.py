"""Generate a tiny synthetic PCB-like VOC dataset for smoke tests.

Not a substitute for the real dataset: it only lets you verify that the whole
pipeline (convert -> train -> export -> serve) runs end to end in minutes.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from src.config import CLASS_NAMES

VOC = """<annotation>
  <filename>{fn}</filename>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
  <object><name>{cls}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>
</annotation>
"""


def _board(w: int, h: int, rng: random.Random) -> np.ndarray:
    img = np.full((h, w, 3), (40, 110, 40), dtype=np.uint8)
    for _ in range(14):
        if rng.random() < 0.5:
            y = rng.randint(20, h - 20)
            cv2.line(img, (10, y), (w - 10, y), (90, 200, 220), 6)
        else:
            x = rng.randint(20, w - 20)
            cv2.line(img, (x, 10), (x, h - 10), (90, 200, 220), 6)
    for _ in range(10):
        c = (rng.randint(30, w - 30), rng.randint(30, h - 30))
        cv2.circle(img, c, 12, (110, 220, 240), -1)
        cv2.circle(img, c, 5, (20, 40, 20), -1)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, 6, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


def _draw_defect(img: np.ndarray, cls: str, cx: int, cy: int, rng: random.Random):
    r = rng.randint(14, 22)
    if cls == "missing_hole":
        cv2.circle(img, (cx, cy), r, (110, 220, 240), -1)
    elif cls == "mouse_bite":
        cv2.circle(img, (cx, cy), r, (0, 0, 0), -1)
    elif cls == "open_circuit":
        cv2.rectangle(img, (cx - r, cy - 8), (cx + r, cy + 8), (40, 110, 40), -1)
    elif cls == "short":
        cv2.rectangle(img, (cx - r, cy - 4), (cx + r, cy + 4), (255, 255, 255), -1)
    elif cls == "spur":
        cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), (255, 60, 60), 3)
    else:  # spurious_copper
        pts = np.array([[cx - r, cy], [cx, cy - r], [cx + r, cy], [cx, cy + r]])
        cv2.fillPoly(img, [pts], (40, 60, 230))
    return cx - r, cy - r, cx + r, cy + r


def generate(out: Path, n_per_class: int = 12, size=(640, 480), seed: int = 0) -> Path:
    rng = random.Random(seed)
    w, h = size
    for cls in CLASS_NAMES:
        (out / "images" / cls).mkdir(parents=True, exist_ok=True)
        (out / "Annotations" / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = _board(w, h, rng)
            cx, cy = rng.randint(60, w - 60), rng.randint(60, h - 60)
            x1, y1, x2, y2 = _draw_defect(img, cls, cx, cy, rng)
            fn = f"syn_{cls}_{i:03d}.jpg"
            cv2.imwrite(str(out / "images" / cls / fn), img)
            (out / "Annotations" / cls / fn.replace(".jpg", ".xml")).write_text(
                VOC.format(fn=fn, w=w, h=h, cls=cls, x1=x1, y1=y1, x2=x2, y2=y2)
            )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("raw_synthetic"))
    ap.add_argument("--n-per-class", type=int, default=12)
    a = ap.parse_args()
    generate(a.out, a.n_per_class)
    print("synthetic VOC dataset at", a.out)
